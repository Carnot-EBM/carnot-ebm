# ARC sp80 execute-stage anomaly: why a "perfect" induction executed to zero progress (2026-07-20)

**Author:** outer-loop diagnostic session (no conductor running; no live-path code modified).
**Trigger:** follow-up to REQ-ARC-WMTE-5726 and its diagnosis note
(`docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md`, §6 first open
question). That note found sp80 was the ONE game whose induction looked genuinely correct
(shard row `sp80` trial 1: `heldout_accuracy=1.0, cell_recall=1.0, goal_predicate_accuracy=1.0`),
yet it executed to `plan_len=1, start_hv=0.0, best_hv=0.0, hv_progress=0.0, reached_levelup=false`.
The open question: **given a genuinely correct induced world model and a plan the system believes
it found, why does executing that plan produce ZERO measured progress?** — smelling like a
plan-finding or plan-execution bug distinct from induction quality.

**Bottom line.** It is NOT a distinct execute-stage bug. There is no plan executor that loses a
genuinely-earned level-up. The anomaly decomposes into two already-known classes plus one
provenance correction, none of which is a fixable executor defect:

1. **`hv_progress=0.0` is a stub-hand_verifier artifact, not a measurement of the plan.** sp80's
   adapter declares `hand_verifier=lambda _game: 0.0` (`arc_game_adapters.py:1494`). The dense
   goal-distance proxy is therefore pinned at 0.0 for sp80 regardless of what any plan does. Three
   other adapters have the same literal-0.0 stub. This is a metric-blindness artifact, not
   evidence a plan made no progress.
2. **The "perfect model" premise is not supported by the current artifacts, and is structurally
   implausible.** The on-disk sp80 engine is a 2048-merge model that is IDENTITY for sp80's real
   actions (4, 5); on the deterministic induction window it scores `heldout=0.0, cell_recall=0.0`
   and `plan_in_model` returns an EMPTY plan. sp80's real L1 solve requires FOUR actions
   (`3×ACTION4 + ACTION5`), so any `plan_len=1` cannot be a correct full solve — it is the induced
   `is_level_complete` firing spuriously early (the model being wrong), which the real env
   correctly does not treat as a level-up.
3. **Provenance correction to the parent note's §4.** The on-disk `world_model.py` this session and
   the parent note both inspected was written 4.5 h AFTER the shard finished, so it is NOT the
   engine the shard scored `heldout=1.0`. The shard's "perfect sp80" engine is overwritten and
   un-inspectable; its 1.0 cannot be reproduced from any surviving artifact.

This confirms the parent note's synthesis (the binding wall is generating a correct world model,
not selecting/searching/executing) and, per the standing framing, is another symptom of the
induction-quality problem — reported plainly rather than forced into a distinct execute-stage
finding.

---

## 1. What the seeded execute path actually does (file:line)

The anomalous row came from `run_seeded_progress`
(`python/carnot/agentic/arc_actions_to_progress.py:641-767`). Its execute sequence:

1. `root_grid = full_traj[0].grid` (`:675`) — the first grid of the offline winning trajectory.
2. `proposer.induce(...)` writes `results/arc_e3/<game>/world_model.py`; `load_engine` reads it
   back (`:702-704`).
3. `plan_in_model(engine, is_done, root_grid, max_nodes=20000, max_depth=40)` (`:707-710`) —
   blind FIFO BFS ENTIRELY INSIDE the induced model (`arc_executable_world_model.py:2009-2135`),
   returning the action sequence the model believes reaches an `is_level_complete` state. Note:
   `run_seeded_progress` calls it with `goal_energy=None`, so it is the original blind BFS, not the
   best-first variant.
4. `_execute_plan_measure(game, plan, hv_fn)` (`:590-638`) — resets a FRESH offline env
   (`env.reset()` + adapter warmup if any), executes the plan's actions, and measures whether a
   REAL level-up fires (`_level_of(frame) > start_level`, the oracle) plus the dense
   `hand_verifier` goal-distance track.

Two facts about this path matter for the anomaly:

- **The dense track reads the adapter's `hand_verifier`.** `hv_fn = _hand_verifier_fn(game)`
  (`:669`, def `:244-267`) returns the adapter's `hand_verifier` or None. `_execute_plan_measure`
  computes `hv_progress = max(0, start_hv - best_hv)/max(|start_hv|,1)` (`:629-631`). If the
  hand_verifier is a constant, `start_hv == best_hv` always and `hv_progress` is pinned at 0.0.
- **The plan is SEARCHED from `root_grid` but EXECUTED from `env.reset()`+warmup.** For sp80
  `warmup_label=None` and `root_grid` is the reset state, so they coincide here — but this coupling
  is a latent fragility for any game where they would not (see §5).

This harness is a DEVELOPMENT MEASUREMENT path, not the scored live agent. Per the ARC Live-Path
Reachability Discipline: `run_seeded_progress` calls `plan_in_model` DIRECTLY (not
`E3AgentPolicy.plan_in_model`), so a defect in *this harness's metric computation* is not a
live-path bug; it would only mislead a development A/B. Any fix here is a harness/metric fix, not a
change to the scored agent's behavior.

---

## 2. Reproduction (deterministic, no LLM needed)

Reused the on-disk `results/arc_e3/sp80/world_model.py` (no re-induction) + the live
`build_progress_window` / `plan_in_model` / `_execute_plan_measure`. `build_window` is fully
deterministic for sp80 (a scripted adapter with fixed labels `SP80_L1_LABELS = 3×ACTION4 +
ACTION5`, unchanged since 2026-06-21). Two independent rebuilds:

```
rep0: cell=1 n_win=4 window_actions=[4, 4, 4, 5] heldout=0.000 cell_recall=0.000 plan_len=0 root_zeros=0
rep1: cell=1 n_win=4 window_actions=[4, 4, 4, 5] heldout=0.000 cell_recall=0.000 plan_len=0 root_zeros=0
```

Full trace of the current on-disk engine on the deterministic window:

```
window transitions:  w0..w2 action=4 (lvl 0->0, changed), w3 action=5 (lvl 0->1, changed)
root_grid (== full_traj[0].grid): 64x64, n_zeros=0, n_nonzero=4096
WorldModelVerifier(window).score(engine):  heldout(exact)=0.0000  cell_recall=0.0000
score_goal_predicate_consistency:          goal_predicate_accuracy=0.7500
is_level_complete(root_grid) = False
plan_in_model(...) -> plan_len=0  (diagnostics: termination_reason='queue_exhausted', nodes_expanded=9)
_execute_plan_measure(...) -> {reached_levelup: False, start_hv: 0.0, best_hv: 0.0, hv_progress: 0.0}
solve_adaptered('sp80',1) -> reached_level=1, solution_labels=['{"action":4}','{"action":4}','{"action":4}','{"action":5}']
```

The current engine scores `heldout=0.0` (not 1.0), yields an EMPTY plan (not `plan_len=1`), and
still reports `hv_progress=0.0`. The `hv_progress=0.0` is invariant to the engine; the `heldout`
and `plan_len` differ from the shard because the on-disk engine is not the shard's engine (§4).

Repro script (scratch): builds the window, scores the on-disk engine, runs `plan_in_model`, and
executes — no GPU/LLM, ~1 min including one offline solve.

---

## 3. Root cause A — `hv_progress=0.0` is a stub-hand_verifier artifact

sp80's adapter (`arc_game_adapters.py:1488-1498`) sets `hand_verifier=lambda _game: 0.0`. It is a
placeholder: sp80 is solved by a fully-scripted `action_labels` list, so the adapter never needed a
goal-distance heuristic, and a non-Optional-friendly call site made a constant lambda the path of
least resistance. Consequences in the harness:

- `_execute_plan_measure` and `run_bounded_progress` both derive `start_hv`, `best_hv`, and
  `hv_progress` from this constant, so all three are pinned at 0.0 for sp80 forever — independent of
  the plan, the engine, or reality.
- The A/B analysis (`paired_by_game(rows, ..., metric="hv_progress")`) treats a stub-0.0 as a real
  measurement and pairs it. The parent note's characterization "the dense hv_progress proxy is 0.0
  for 9 of the 11 plan_found rows" is therefore partly an ARTIFACT: some of those 0.0s are stub
  games, not real zero-progress observations.

Four adapters carry a literal-0.0 stub hand_verifier:

| Game | Line | Stub |
|---|---|---|
| su15 | `arc_game_adapters.py:1330` | `lambda _game, _frame=None: 0.0` |
| sp80 | `arc_game_adapters.py:1494` | `lambda _game: 0.0` |
| cn04 | `arc_game_adapters.py:1554` | `lambda _game, _frame=None: 0.0` |
| ka59 | `arc_game_adapters.py:2213` | `lambda _game, _frame=None: 0.0` |

So for these games the harness's dense discriminator silently reports "no progress" when the honest
statement is "no dense progress signal available." This does not affect `reached_levelup` (the
sparse oracle), which remains correct.

---

## 4. Root cause B — the "perfect model" premise is not supported by surviving artifacts

**Provenance.** The shard JSONL `results/exp5726_thinkingcap_16k_dualgpu_shard.jsonl` was last
written `2026-07-19 14:11:12` (all rows, including sp80, predate this). The on-disk
`results/arc_e3/sp80/world_model.py` was written `2026-07-19 18:47:48` — 4.5 h LATER. Because
`run_seeded_progress` deletes-then-rewrites a per-GAME `world_model.py` on every induce
(`:696-702`), the on-disk file is the LAST successful sp80 induction across ALL runs, which here is
a later run, NOT the shard cell that recorded `heldout=1.0`. The shard's "perfect" engine is
overwritten and un-inspectable. This SHARPENS the parent note's §4 attribution caveat: its implicit
identification of "sp80's on-disk 2048 model" with "the heldout=1.0 row" is not supported — the
inspected engine (which THIS session shows scores 0.0) postdates the shard.

**Structural implausibility of a correct one-action sp80 solve.** sp80's real L1 needs four actions
(`3×ACTION4` to move the splitter right, then `ACTION5` to commit the spill —
`solve_adaptered` returns exactly `[4,4,4,5]`). The induced engine class is a 2048-merge model
whose action map is `{0:up,1:down,2:left,3:right}` and returns `grid` unchanged for any other action
(`world_model.py:23-24`) — i.e. IDENTITY for sp80's real actions 4 and 5. Therefore:

- A 2048-identity engine cannot reproduce a state-CHANGING sp80 transition (it predicts no change),
  which is why the current engine scores `heldout=cell_recall=0.0` on the genuine `[4,4,4,5]`
  window. A shard engine that scored `heldout=1.0` AND `cell_recall=1.0` on such a window is only
  possible against a materially different window (e.g. one whose transitions were no-ops, or used
  actions 0-3) — i.e. a window/engine attribution artifact, not a correct sp80 dynamics model.
- A `plan_len=1` plan cannot be a correct sp80 L1 solve (which needs 4 real actions). It is the
  induced `is_level_complete` accepting a state one action from root that the real env does not
  treat as a level-up. `_execute_plan_measure` correctly reports `reached_levelup=False`.

So `reached_levelup=False` is the CORRECT outcome of executing a plan built on an incorrect model —
not a bug that loses a real win.

---

## 5. Latent fragility worth noting (not fixed here)

`plan_in_model` searches from `root_grid = full_traj[0].grid`, but `_execute_plan_measure` executes
from `env.reset()` + adapter warmup (`arc_actions_to_progress.py:602-605`). For sp80
(`warmup_label=None`, `root_grid` == reset state) these coincide, so it is not the sp80 bug. But for
any game where `full_traj[0]` is NOT the reset+warmup state (e.g. a game whose winning trajectory the
window is built from starts mid-episode, or that has a non-trivial warmup the window build applies
but `_execute_plan_measure` applies differently), the plan would be searched for one starting state
and executed from another, producing spurious `reached_levelup=False`. This coupling is a review
item for a future faithfulness pass on the seeded harness, not an active sp80 defect.

---

## 6. Recommendation (operator-only whether to act)

No code changed this session. The one narrow, well-understood defect (Root cause A) has two possible
fixes, both with a caveat that keeps them out of a "clearly-safe, minimal" auto-apply this task:

- **Option 1 (adapter): replace the four `lambda …: 0.0` stubs with `None`.** `None` is the existing
  sentinel for "no hand_verifier"; the harness already handles it correctly (`_hand_verifier_fn`
  returns None → `hv_progress` stays None → `paired_by_game` excludes the game from the hv metric,
  the honest outcome). **Caveat / why not auto-applied:** `arc_loop_solve.py:62-63`
  (`_live_verifier_for_adapter`) uses `adapter.hand_verifier` as the COLD-START best-first search
  heuristic fallback when no spatial/learned checkpoint exists. Setting it to `None` could break a
  future cold-start solve of these four games (a downstream caller may `verifier(state)` on None).
  Requires validating that all four games still reproduce offline (checkpoint-warm AND cold-start)
  before shipping — a 4-game solve A/B, wider than this task's budget.
- **Option 2 (harness): make the metric honest without touching adapters.** Track distinct
  hand_verifier values observed per run; expose `hv_informative` (≥2 distinct values) and have the
  hv-metric A/B (`paired_by_game`/`paired_summary` when `metric` is hv-derived) exclude
  non-informative rows. Safe w.r.t. OfflineSolver (adapters untouched), but changes existing A/B
  outputs for the hv_progress metric (removes fake-0.0 pollution — the correct direction, but a
  behavior change that should be reviewed before landing, and it risks nulling a genuinely-flat but
  real hv run, losing a real 0.0 observation).

Neither is a plan-executor fix, because there is no plan-executor bug. The higher-leverage target
remains the parent note's conclusion: generating a correct world model (correct action semantics for
the game), which for sp80 means an engine class that models splitter-placement + spill-commit rather
than 2048 merges.

---

## 7. Cross-references

- `docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md` — parent note
  (§4 attribution caveat this note sharpens; §6 open question this note answers).
- `python/carnot/agentic/arc_actions_to_progress.py:641-767` (`run_seeded_progress`), `:590-638`
  (`_execute_plan_measure`), `:244-267` (`_hand_verifier_fn`).
- `python/carnot/agentic/arc_executable_world_model.py:2009-2135` (`plan_in_model`), `:702-752`
  (`WorldModelVerifier.score`).
- `python/carnot/agentic/arc_game_adapters.py:1330/1494/1554/2213` (the four stub hand_verifiers),
  `:1442-1498` (`_sp80` adapter + the `[4,4,4,5]` scripted L1 solve), `:181` (`SP80_L1_LABELS`).
- `scripts/arc_loop_solve.py:44-63` (`_live_verifier_for_adapter` — the cold-start hand_verifier
  dependency that makes Option 1 not clearly-safe).
- CLAUDE.md "ARC Live-Path Reachability Discipline" (why a harness-metric fix is not a live-path
  change), "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" (an offline null can be a
  harness/corpus artifact — realized here for both the hv metric and the shard-engine provenance).
