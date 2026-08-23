# Do broken goal predicates harm the live agent, and where?

Date: 2026-08-23. Method: read-only over `results/**` plus an in-process re-run
of the real planner on the emitted models. No GPU. No generator. No live game.

## The question

An earlier review settled that the scored agent does NOT use the induced
`is_level_complete` to recognise a win. Level detection comes from the
environment (`arc_agi3_live_adapter.py:352`). See
`goal-predicate-flat-rate-adversarial-review-2026-08-23.md`.

The predicate still has two real uses, both inside search:

1. `plan_in_model` (`arc_executable_world_model.py:8526`) uses it as the
   TERMINAL TEST for a BFS inside the induced model, at zero real actions.
2. `_install_goal_bias` (`arc_competition_agent.py:5980`) uses it to order the
   best-first frontier.

This note measures what a broken predicate does at each site, and how often.

## Answer in one paragraph

A broken predicate is caught, never crashes anything, and silently makes the
internal BFS non-terminating-by-construction. Where the predicate can bind it
does bind: predicates that provably cannot report a win find ZERO plans, against
36% for the rest (Fisher p=0.005). But the predicate is NOT the binding
constraint. In 58% of the 819 plan searches on record, the induced ENGINE's
reachable set collapsed to the root before the predicate was consulted even
once. And at the goal-bias site the mechanism was never contributing anyway: it
scored degenerate in 32 of 32 recorded runs, and a perfect predicate would have
scored the same.

## 1. Failure behaviour, per call site

### `plan_in_model` — caught, no abort, full budget burned

The predicate runs at `:8676` (goal-energy branch) and `:8749` (FIFO branch).

| what the predicate does | what happens | code |
|---|---|---|
| raises a plain `Exception` | caught by `except Exception: pass`; the state is then ENQUEUED anyway and treated as non-terminal | `:8696-8701`, `:8765-8767` |
| raises `EngineCallGuardError` (hang or RSS growth) | counted a trip, state skipped; at `guard_max_trips()` the whole search aborts with `termination_reason="engine_guard_tripped"` | `:8687-8695`, `:8758-8764` |
| returns False always | `bool(...)` is False, so the search never returns | same lines |
| has no `return` at all | returns `None`; `bool(None)` is False; identical to the row above | same lines |

So the search does not fall back, does not abort, and does not crash the cell.
It explores until `max_nodes` or until the frontier drains, then returns `None`.
`_termination_reason` (`:8423`) labels it `max_nodes_reached`, `depth_capped` or
`queue_exhausted`.

**All three broken classes are indistinguishable in the diagnostics.** The
function counts `nodes` (engine calls), `depth_truncated_nodes` and
`engine_guard_trips`. It counts NO predicate evaluations and NO predicate
exceptions. So `queue_exhausted` after zero predicate calls, and
`queue_exhausted` after 5000 of them, are the same string. Those two mean
opposite things — a dead engine versus a dead goal — and they lead to opposite
fixes. This is the same conflation `_termination_reason`'s own docstring was
written to remove on the depth axis, still open on the predicate axis.

One exposure worth naming: `except Exception` does not catch `SystemExit` or
`KeyboardInterrupt`. Neither does any caller (`arc_competition_agent.py:7330`
catches `Exception`). A generated predicate raising one of those would escape
every handler. Not observed; recorded because it is cheap to record.

What the callers do with `None`:

- `:7327` carried goal -> `reason="no_plan_in_carried_model"`. The goal bias is
  NOT installed; `:7341` fires only when a plan was found.
- `:7528` ttt-prior -> stores `ttt_prior_engine_plan_diagnostics`. It STILL
  installs the bias unless the goal-energy search showed no improvement
  (`:7552-7564`).
- `:8309`/`:8313` plain path -> the bias is installed at `:8309` BEFORE planning
  runs, so a broken predicate is always installed here. On `None` the agent
  falls through to the subgoal and factored planners.
- `arc_llm_reinduction.py:2195` -> `skipped="no_reachable_plan_after_refinement"`
  (`:1649`).

### `_install_goal_bias` — a broken predicate and a working one are the same

The default binary `_bias` (`:6025-6030`) has no try/except of its own. Its only
consumer does: `_goal_bias_score` (`:2477-2504`) wraps the call, increments
`self._goal_bias_errors`, and returns 0.0 (`:2502-2504`).

A raising predicate scores 0.0. An always-False predicate scores 0.0. **A
CORRECT binary predicate also scores 0.0 at every non-terminal node** — the
"CLIFF" the method's own docstring names at `:5983-5987`. So this site cannot
tell a broken predicate from a working one except at an exact win state.

The graded path (`CARNOT_ARC_GRADED_GOAL_BIAS=1`) would differ, but it is off by
default and never fired in the record (see section 3e).

## 2. The live population, classified behaviourally

Population: the guarded scorer's default root
(`scripts/arc_induction_quality.py`, REQ-ARC-WMTE-6700). 114 `world_model.py`
files kept; 1141 excluded as `nested_repo_clone`. 93 distinct files, 88 distinct
predicates.

Each model was executed in its own subprocess against its own game's REAL
logical root grid, taken from the offline arcade. Then the real `plan_in_model`
ran on it at the production budget (`max_nodes=20000`), with the predicate
wrapped in a counting proxy. 110 of 114 probed; 4 are scratch directories
(`g`, `g1`, `g3`) with no arcade game.

| class | files (n=110) | distinct predicates (n=87) |
|---|---|---|
| raises on real input | **0** | **0** |
| cannot report a win: every `return` is a falsy constant | 15 (13.6%) | 10 (11.5%) |
| no `return` statement at all | 0 | 0 |
| returns True on the level's own root grid | 3 | 3 |
| `is_level_complete` absent | 4 files | -- |
| no defect detected | 92 | 74 |

**Zero predicates raised.** Not on the root grid, and not on any of up to 5858
in-search evaluations per model. The premise that ~8% raise at runtime does not
reproduce on the live population under real inputs. The only recorded raise in
the whole evidence tree is `NameError("name 'numpy' is not defined")` on lp85
(`results/arc_engine_retention_20260729/cells/ret1__lp85__s1.json`) — a missing
import, not an array used as a truth value.

The cannot-win class DOES reproduce, at 13.6% of files and 11.5% of distinct
predicates.

Consistency check on the classifier: all 15 cannot-win files found no plan, as
they must. Nothing in the class contradicts its own definition.

## 3. Is the harm visible in run artifacts?

### a. The row-level artifacts cannot answer it

`scripts/arc_scored_path_lever_harness.py:796-843` reduces
`policy.induction_attempts` to counters. It carries `induction_planned`,
`induction_skipped`, `induction_reasons`, `induction_exceptions`. It carries
NONE of `plan_diagnostics`, `ttt_prior_engine_plan_diagnostics`,
`goal_satisfiability`, `goal_predicate_satisfiable`. So `baseline25/rows.json`
and its siblings record THAT a plan search failed and never WHY.

That absence is itself the finding for those files.

### b. The committed evidence tree can

819 real `plan_in_model` diagnostics blobs, from 18807 JSON files under
`results/`. Matched on `termination_reason` beside a key the planner always
writes, so an unrelated field of the same name is not counted.

| termination_reason | n | share |
|---|---|---|
| queue_exhausted | 415 | 50.7% |
| max_nodes_reached | 239 | 29.2% |
| plan_found | 130 | 15.9% |
| depth_capped | 30 | 3.7% |
| is_level_complete_none | 5 | 0.6% |

### c. The modal search never leaves the root

`nodes_expanded` for `queue_exhausted`: median 37, and the MODE is exactly 37
with 107 occurrences. `_model_candidates` (`:8398-8420`) returns 5 keyboard
actions plus up to 32 clicks — at most 37 per node.

240 of 415 `queue_exhausted` searches (58%) ended at 37 engine calls or fewer.
They expanded the ROOT NODE ONLY and then the frontier was empty: every
successor was shape-mismatched or already in `seen`. The induced engine
predicted that no action changes anything.

The in-process re-run confirms the mechanism directly. **21 of 110 predicates
were never evaluated once** during a full 20000-node search, at
`nodes_expanded` between 12 and 37. For those models the predicate could have
been perfect or absent and the outcome would be byte-identical.

Worked example, `baseline25/e3/sp80/world_model.py`: `nodes_expanded=12`,
`termination_reason=queue_exhausted`, predicate evaluations = 0.

### d. Where the predicate CAN bind, it does

| population | cannot-win: plans found | rest: plans found | Fisher exact |
|---|---|---|---|
| files (n=110) | 0 / 15 | 34 / 95 (35.8%) | p = 0.0048 |
| distinct predicates (n=87) | 0 / 10 | 28 / 77 (36.4%) | p = 0.027 |

This is a real effect and it is not the dominant one. Both facts hold together:
a cannot-win predicate zeroes out planning for its own model, and most models
never get far enough for that to be what stopped them.

### e. The goal-bias site contributes nothing, broken or not

72 `goal_bias_diagnostics` blobs in `results/`; 34 enabled; 32 scored at least
one node.

- `errors > 0` in **ZERO** of the 32. No predicate raised at this site, ever, on
  the record.
- `degenerate: true` in **32 of 32**. `score_variance == 0.0` in 32 of 32.

A goal bias returning the same constant at every node cannot influence frontier
ordering. That is the self-audit at `:2298-2325` firing at 100%.

Only 3 of the 32 carry an `L*_induced_goal_predicate` label; 29 are
`exp4020_graded_goal_satisfaction_energy`, a different energy source. So n=3 for
the induced predicate specifically. State that as the small number it is.

The planner-side energy agrees. Of 114 searches recording a `goal_energy_source`:
112 `binary`, 2 `novelty`, **0 `graded_exemplar`**. The binary source returns a
constant at every non-goal state. So the frontier ordering had no gradient in
essentially every recorded search.

### f. Real harm, from the OTHER degeneracy

Three live models return True on their own level's root grid. That is not a
predicate that cannot win; it is one that thinks it already has.

| model | plan | nodes_expanded | predicate calls |
|---|---|---|---|
| `baseline25/e3/g50t/world_model.py` | found, length 1 | 2 | 1 |
| `baseline25/e3/sb26/world_model.py` | found, length 2 | 65 | 8 |
| `repair_widen/e3_w4b/sb26/world_model.py` | found, length 2 | 34 | 5 |

g50t's own row in `baseline25/rows.json` reads `induction_planned=1`,
`levels=0`. The agent planned, executed a one-action "win", and banked nothing.

`_goal_satisfiability_check` exists to catch exactly this and returns
`goal_predicate_true_at_root` (`arc_llm_reinduction.py:837-848`). That kind
appears **zero** times in the entire evidence tree, against 20 recorded
`degenerate_goal_predicate`. On the plain path the gate is behind
`CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK`, unset in production
(`arc_competition_agent.py:8269`).

Only 22 goal-gate verdicts are recorded anywhere, against 819 plan searches. The
gate's opinion is almost never persisted.

## 4. The distinction the question turned on

"No harm" and "the mechanism was never contributing" look identical in level
counts. They are different here, and both are present, at different sites.

- **Goal bias: the mechanism was never contributing.** Degenerate in 32 of 32
  recorded runs, with a constant energy in 112 of 114. Fixing every predicate
  would not move it, because the binary bias is a cliff by construction. This
  site needs a redesign or a retirement, not better predicates.
- **Planner: the predicate matters but is not the bottleneck.** It zeroes out
  planning when it cannot win (p=0.005), and in 58% of recorded searches the
  engine's reachable set had already collapsed to the root.

The ordering that follows: engine quality first, root-true degeneracy second,
cannot-win predicates third, goal-bias design separately.

## 5. What I could not determine

- **Whether any specific historical plan failure was caused by its predicate.**
  `results/arc_e3/<game>/world_model.py` is overwritten in place on every
  refinement round, so today's predicate is not necessarily the one that ran.
  Section 2 measures the CURRENT population; section 3b-3c measures HISTORICAL
  searches. They are not joined, and cannot be joined from what is on disk.
- **How often a predicate raises in a live episode.** `_goal_bias_errors` is the
  only counter and reads 0 across 32 scored runs, but only 3 of those carry the
  induced-predicate label.
- **How often `plan_in_model` is reached on the scored path.** The row schema
  does not record the number of searches, only `induction_planned`.
- **Whether the 15 cannot-win files were ever used live**, or are discarded
  refinement rounds left on disk.
- **Whether array-truthiness raises exist at all.** Zero appeared under real
  inputs. A static reading of the source may still find the pattern on a code
  path my probe did not reach.

## 6. Step 6: check, not prose

Four gaps. None is built here; this note is a measurement.

1. **`plan_in_model` must count predicate evaluations and predicate
   exceptions**, beside `engine_guard_trips`. Today a search that never called
   the predicate and a search that called it 5000 times report the same
   `queue_exhausted`. Cheap, and it fires on no legitimate work.
2. **`_termination_reason` should split "frontier drained with zero predicate
   evaluations" from "frontier drained after evaluating N states."** The first
   is a verdict on the ENGINE. The second is a verdict on the GOAL. This is the
   depth-axis fix repeated on the predicate axis.
3. **The harness should carry `plan_diagnostics` per attempt.** It already
   learned this lesson twice, for `induction_exceptions` and
   `induction_proposer_notes`. The same tally-without-the-message shape is still
   in place for plan searches.
4. **Check `goal_predicate_true_at_root` on the plain path.** Three live models
   are true at root, one of them planned a bogus one-action win in a real
   baseline run, and the record holds zero verdicts of that kind. Note the gate
   is default-off for a stated reason (a measured false-negative risk); the
   root-true check specifically is the cheap, non-search half of it and does not
   carry that risk.

## Reproduction

Scripts are in `$CLAUDE_JOB_DIR/tmp/gp_probe/` (scratch, not committed):
`cache_roots.py` (offline-arcade root grids), `probe_one.py` (one model, one
subprocess), `sweep.py`, `extract_plan_diag.py`, `extract_goal_bias.py`,
`extract_goal_gate.py`. All read-only with respect to `results/**`.

One trap worth recording: the first sweep set `RLIMIT_AS` to 6 GB before
importing the planner. Torch reserves tens of GB of VIRTUAL address space
without touching the pages, so all 110 probes hung at import and reported
TIMEOUT. A timeout that is really an import failure looks exactly like a slow
engine. Import first, then set the limit.

## Cross-references

- `docs/research-notes/goal-predicate-flat-rate-adversarial-review-2026-08-23.md`
  -- the review that established the env supplies win detection
- `python/carnot/agentic/arc_executable_world_model.py:8526` -- `plan_in_model`
- `python/carnot/agentic/arc_executable_world_model.py:8398` --
  `_model_candidates`, the source of the 37
- `python/carnot/agentic/arc_competition_agent.py:5980` -- `_install_goal_bias`
- `python/carnot/agentic/arc_competition_agent.py:2477` -- `_goal_bias_score`,
  the only place a predicate exception is counted
- `python/carnot/agentic/arc_llm_reinduction.py:698` --
  `_goal_satisfiability_check`
- `scripts/arc_scored_path_lever_harness.py:796` -- the row aggregation that
  drops plan diagnostics
- CLAUDE.md "ARC Live-Path Reachability Discipline" -- the two live entrypoints
- CLAUDE.md "The Error Lifecycle" step 6 -- why section 6 names checks
