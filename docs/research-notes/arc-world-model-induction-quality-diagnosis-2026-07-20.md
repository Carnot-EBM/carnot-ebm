# ARC world-model induction-quality diagnosis: heldout_accuracy near-zero (2026-07-20)

**Author:** outer-loop diagnostic session (no conductor; no live-path code modified).
**Trigger:** REQ-ARC-WMTE-5726 (`results/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.json`
+ raw per-cell shard `results/exp5726_thinkingcap_16k_dualgpu_shard.jsonl`). ThinkingCap-27B
beat Qwen-9B on `induce_ok` completions (31/40 vs 6/40, Wilcoxon p=0.0005), but ALL 80 cells
(both arms) banked ZERO level-ups, and the 31 "successful" ThinkingCap completions carry a
near-universal `heldout_accuracy ≈ 0.0` even when `cell_recall` and `goal_predicate_accuracy`
are substantial.

**This is a diagnosis, not a fix.** No live-path code was changed. The one-line synthesis is in
§5; a strictly-separated "open questions for a follow-up" list is in §6. Every quantitative
claim below is a real count from the 80-row shard; every code claim carries a `file:line`.

---

## 1. Metric definitions, precisely (why they can disagree)

All four numbers are computed inside `run_seeded_progress`
(`python/carnot/agentic/arc_actions_to_progress.py:641-767`) against the SAME level-up-straddling
`window` (the transitions from one level boundary, built by `build_progress_window` →
`experiment_5717.build_window`). They disagree because each measures a **different quantity over a
different subset with a different aggregation**:

| Field | Defined at | What it measures | Aggregation |
|---|---|---|---|
| `heldout_accuracy` | `arc_executable_world_model.py:711-752` (`WorldModelVerifier.score` → `VerifyResult.accuracy`) | **Exact FULL-GRID byte match**: a transition counts as correct only if `pred.shape == next_grid.shape AND np.array_equal(pred, next_grid)` (`:733-734`), i.e. every cell of the predicted grid equals the true next grid. | `n_correct / n` over **ALL** window transitions (changing AND no-op), `:752`. |
| `cell_recall` | `arc_executable_world_model.py:726-732, 751` | **Soft changed-cell recall**: for a state-CHANGING transition, mask `m = (grid != next_grid)`; score `(pred[m] == next_grid[m]).mean()` — the fraction of the cells that *truly changed* that the engine predicted right. | `np.mean(...)` over **CHANGING transitions ONLY** (`:727` gates on `changed`), `:751`. |
| `goal_predicate_accuracy` | `arc_executable_world_model.py:847-896` (`score_goal_predicate_consistency`) | Sign agreement: for each transition, `is_level_complete(next_grid) == (level_after > level_before)` (`:887`). | `n_correct / n` over ALL window transitions, `:896`. |
| `levelup_positive_recall` | `arc_actions_to_progress.py:563-577` | Positive-class recall of the goal predicate: of the REAL level-up transitions only, the fraction whose `next_grid` `is_level_complete` recognizes as a win (`:571-575`). | `hits / total_real_levelups`, `None` if the window has no real level-up. |

Three structural reasons these disagree — all load-bearing:

1. **`heldout_accuracy` is exact-match; `cell_recall` is soft partial-credit.** An engine that gets
   the *direction/location* of change mostly right but is not byte-perfect (off by a few cells, a
   slightly-wrong recolor value, or a spurious write elsewhere) scores `cell_recall` 0.3–0.9 but
   `heldout_accuracy = 0.0`. The `VerifyResult` docstring names this exact effect: "exact-match
   reads ~0 for an imperfect … world model that is still ~55 % changed-cell-accurate"
   (`:694-698`).

2. **`cell_recall` only looks at cells that truly changed, and does NOT penalize spurious writes to
   cells that should have stayed put.** The mask `m` is the *true* change-set (`:729`); the engine
   is never scored on the untouched region. So an engine can have `cell_recall = 0.91` while its
   full grid is wrong (it also changed cells it shouldn't have, or got the exact values slightly
   off) → `heldout_accuracy = 0.0`.

3. **`goal_predicate_accuracy` is dominated by the majority no-op class.** The window is mostly
   non-level-up transitions, so a predicate that *never* claims a win scores high accuracy by
   correctly saying "not a win" for every no-op — while contributing nothing. `levelup_positive_recall`
   is the imbalance-corrected companion that exposes this (see §3).

Empirically, over the 37 `induce_ok` rows with metrics, the three are near-independent:
Pearson `r(heldout, cell_recall) = 0.549` (weakly related — same phenomenon, different strictness),
`r(heldout, goal_pred) = −0.041`, `r(cell_recall, goal_pred) = 0.068`. **A high
`goal_predicate_accuracy` tells you essentially nothing about dynamics accuracy.**

---

## 2. The systematic pattern across all 80 cells

Shard: 80 rows = 40 ThinkingCap-27B + 40 Qwen-9B, all `arm_kind=reason`.

**Completion (`induce_ok` = the LLM emitted valid `engine` + `is_level_complete` code):**
- ThinkingCap: 31/40. Qwen: 6/40. This reproduces the headline win.
- The 34 Qwen failures are **code-emission** failures, not model-quality failures: 33/34 have
  `last_stop_type="limit"` + `overran=True` (hit the `n_predict=16384` output cap before closing
  out valid code), 1 `eos`. ThinkingCap's 9 failures are the same class (9/9 `limit`+`overran`).
  So ThinkingCap's edge is that it **finishes writing the program** more often — it is upstream of,
  and orthogonal to, whether the finished program is *correct*.

**Quality of the 37 completions that DID induce (31 TC + 6 Qwen), all with metrics populated:**

| Statistic | Value |
|---|---|
| `heldout_accuracy` min / median / mean / max | 0.0 / **0.0** / 0.124 / 1.0 |
| `cell_recall` min / median / mean / max | 0.0 / 0.369 / 0.394 / 1.0 |
| `goal_predicate_accuracy` min / median / mean / max | 0.0 / **1.0** / 0.919 / 1.0 |
| rows `heldout_accuracy == 0.0` | **29 / 37** |
| rows `heldout == 0.0` AND `cell_recall > 0.5` | **9 / 37** (captures ≥½ the changed cells, never a full transition) |
| rows `heldout == 0.0` AND `cell_recall > 0.3` | 16 / 37 |
| rows `heldout < 0.5` (would FAIL the live default exact gate) | **33 / 37** |
| rows `heldout < 0.5` AND `cell_recall >= 0.5` | 9 / 37 (admitted by a cell_recall gate, rejected by the default) |
| rows `goal_predicate_accuracy >= 0.875` | 33 / 37 |
| rows `goal_predicate_accuracy == 1.0` | 19 / 37 |

Restricting to ThinkingCap's 31 completions: only **4** reach `heldout ≥ 0.5` (games `ft09`,
`sp80`, `su15`), 4 land in `(0, 0.5)`, and **23/31 (74 %) score `heldout = 0.0`** — never once
reproduce a full transition exactly. The distinct games covered by TC's completions: ar25, cd82,
ft09, g50t, lf52, lp85, ls20, m0r0, r11l, re86, sb26, sk48, sp80, su15, tr87, tu93, vc33.

**Planning and progress (the downstream floor):**
- `plan_found`: 11/37 (all ThinkingCap; Qwen 0). Of those 11, **6 have `heldout = 0.0`** and
  **9 have `heldout < 0.5`** — i.e. 9/11 plans were built inside a world model the live default
  gate would reject. Only 2 (`sp80` t0/t1) would clear `heldout ≥ 0.5`.
- **`reached_levelup`: 0 / 80.** Every plan that was found and executed against the real env
  reached ZERO real level-ups. The dense `hv_progress` proxy is 0.0 for 9 of the 11 plan_found
  rows; the only non-zero movement is `ar25` t1 (0.333) and `re86` t0 (0.055).
- Plan lengths are almost all `plan_len=1` (cd82, r11l, re86, sb26, sp80 …) — the mark of a
  goal predicate that fires after a single action (§3, §4).

---

## 3. The trust-gate check (GAP-WM-TRUST-GATE)

`ops/verifier_gaps.md:2686-2713` (GAP-WM-TRUST-GATE) documents that the live
`_induce_and_plan` trust gate `accuracy >= 0.5` can be **gamed by an identity engine** on
no-op-heavy click games: a `return grid` engine reproduces every no-op and clears 0.5 while
predicting no change, so `plan_in_model` finds no path and the tier contributes nothing.

**What the live gate actually is.** The plain single-shot path
(`arc_competition_agent.py:4203-4216`) computes `vr = WorldModelVerifier(active_transitions).score(engine)`
and gates on `_gate_value = vr.cell_recall if CARNOT_ARC_TRUST_METRIC=="cell_recall" else vr.accuracy`,
rejecting when `_gate_value < 0.5` (`:4214`, `skipped="world_model_accuracy_below_threshold"`). The
env var **defaults to `"exact"`** (`:4209`), so the shipped gate is **`heldout_accuracy ≥ 0.5`**.

**Can this shard confirm or refute the gate concern? Partially — and the honest answer has three
parts:**

1. **The shard's `plan_found` does NOT test the live gate.** The seeded harness plans *ungated*:
   `run_seeded_progress` calls `plan_in_model` whenever `engine is not None and is_done is not
   None` (`arc_actions_to_progress.py:707-710`) — there is **no `accuracy >= 0.5` check** between
   induce and plan. So "9/11 plan_found rows have `heldout < 0.5`" is a property of the *harness
   bypassing the gate*, NOT evidence that the live gate admits weak models. This must not be
   over-read.

2. **For the transition ENGINE, the shipped gate would REJECT these models, not admit them.**
   33/37 completions have `heldout < 0.5`, so under the default `exact` gate the induce→plan tier
   would fire `skipped="world_model_accuracy_below_threshold"` and contribute nothing. This is the
   **opposite** of the identity-admission failure GAP-WM-TRUST-GATE describes — it is the
   *false-negative wall* the `VerifyResult` docstring (`:694-698`) and the gate comment
   (`:4204-4207`) both name: exact-match gates out imperfect-but-nonzero models. GAP-WM-TRUST-GATE's
   identity-admission scenario (identity scoring ≥0.5 on a no-op-heavy corpus) is a *possible*
   corpus (the `build_window` windows do contain many no-ops), but it is **not the dominant
   phenomenon in this data** — these ThinkingCap/Qwen engines fail low, they do not pass high.

3. **The identity-gaming phenomenon IS empirically present here — transposed onto the GOAL
   predicate.** 18/37 rows have `levelup_positive_recall = 0.0` (the induced `is_level_complete`
   NEVER recognizes a single real win) yet a mean `goal_predicate_accuracy` of **0.833** — 16 of
   those 18 sit at exactly **0.917** (consistent with a 12-transition window holding 1 real
   level-up: a constant-`False` predicate scores 11/12 = 0.917). This is precisely the
   class-imbalance gaming GAP-WM-TRUST-GATE warns about, but it inflates the **goal-predicate**
   score rather than the engine score. `goal_predicate_accuracy` is therefore an actively
   misleading trust signal on these windows; `levelup_positive_recall` is the honest one.

**Would the GAP's proposed fix have excluded these models? Undeterminable in the exact form the
GAP specifies, answerable in spirit:**
- The GAP proposes `change_accuracy >= T AND n_changes_correct >= k`. The shard captures
  `cell_recall` and `heldout_accuracy` per row, but **NOT** the raw per-transition predictions,
  `n_changes_correct`, or a full-transition `change_accuracy` (exact-match restricted to changing
  transitions). So the GAP's exact two-part predicate **cannot be recomputed from these
  artifacts** — only qualitatively reasoned about.
- Qualitatively: `cell_recall` *is* the change-weighted quantity the GAP wants. Switching the live
  gate to `cell_recall` (`CARNOT_ARC_TRUST_METRIC=cell_recall`) would **admit 9/37 rows the exact
  gate rejects** (`heldout < 0.5, cell_recall ≥ 0.5`). It also **defends against the identity
  direction** the GAP feared: an identity engine predicts no change, so on every changing
  transition all changed cells are wrong → `cell_recall = 0`, well below 0.5. So the change-weighted
  gate simultaneously (a) removes the exact-match false-negative wall and (b) closes the
  identity-admission hole. **But it does not fix quality:** all 11 plans (several from
  `cell_recall ≥ 0.5` engines like `sp80` t0 `cr=0.978`) reached 0 real level-ups. A better gate
  changes *which wrong models plan*, not *whether the plans work*.

---

## 4. What the induced code actually is (concrete inspection)

**Attribution caveat (stated honestly).** The shard does not embed the induced source. The only
induced code available is the on-disk `results/arc_e3/<game>/world_model.py`, which `load_engine`
reads from a **per-GAME path shared across arms/trials** and which `run_seeded_progress` deletes +
rewrites on every induce (`arc_actions_to_progress.py:696-702`). So each on-disk file is the **last
*successful* induce for that game**, and cannot be mapped to a specific shard row. Two files
(`cd82` @ 07-20 10:41, `g` @ 07-20 10:24) were overwritten by a *later* run today and are excluded
from the shard sample; the rest are stamped 2026-07-19 18:xx (the shard run). Read the code below as
a **qualitative census of what these engines look like**, not per-row ground truth. Where an on-disk
predicate contradicts a shard row (e.g. on-disk `r11l` has `is_level_complete → False` but shard
`r11l` t0 had `lpr=1.0`), that is exactly the attribution gap — a *different* trial's engine is on
disk.

Six recurring failure patterns, all present in the 07-19 sample:

1. **Pure identity engine** (`re86`, and the shard's near-identity `cd82`): `return grid` for every
   action. Models inaction. This is the engine class GAP-WM-TRUST-GATE flagged.
2. **Single hardcoded click-recolor** (`r11l`: `if action==6: grid[py,px]=5`; `lp85`: recolor
   `14→15` on click): one guessed effect, no real mechanic.
3. **Confabulated plausible-but-wrong mechanic** (`sb26`: invents a "flood the 3×3 neighborhood to
   color 4" rule, self-commented *"This is a simplified rule based on the observed transitions"* —
   the LLM narrating its own guess; `vc33`: hallucinates a Sokoban movement model with a wrong,
   duplicated action map `0=up…5=up…9=stay` and mislabels colors as `goal_reached`). Structured,
   coherent-looking, and wrong.
4. **Window memorization / overfitting** (`ls20`): the engine **hardcodes literal pixel coordinates
   copied from the observed window** — *"ACTION 1: Place a block at (61, 13)"*, *"ACTION 3: Place a
   block at rows 45-49, cols 29"*. This is the mechanism behind *high cell_recall + zero
   heldout_accuracy*: memorized coordinates reproduce the changed cells of the very transitions they
   were fit to (recall), but generalize to nothing on held-out transitions (exact = 0).
5. **Degenerate goal predicates**, three sub-shapes:
   - `return False` (`r11l`, `re86`, near-identity `cd82`) → never fires → `levelup_positive_recall=0`.
   - `return True` (`ls20`, a self-described *"placeholder implementation"*) → fires on every state
     → spurious `plan_found` with `plan_len=1`.
   - Whole-grid-uniform `np.all(grid == C)` (`sb26`: `==4`; `ar25`: no-`9`s remain; `lp85`:
     `==15`) → a strict condition that is almost never true in reality, or trivially "true" only
     after the model paints the whole board.
   This is why `plan_found` is not a quality signal: `plan_in_model` returns a plan **iff the
   induced `is_level_complete` is satisfiable from `root_grid` within the search bound**
   (`arc_executable_world_model.py:2019-2052`) — a degenerate predicate makes that trivially true,
   usually at `plan_len=1`, and the plan then does nothing in the real env.
6. **Correct only when the mechanic matches a game template the LLM already knows.** `sp80` is a
   clean, coherent **2048-style merge model** (`process_line` strips zeros, merges equal adjacent
   tiles ×2; `is_level_complete` = board full with no adjacent equals) — and `sp80` is exactly where
   `heldout` hit **1.0** (t1) and 0.5 (t0). `ft09` (heldout 1.0) and `su15` (0.714) are the other
   two template-matched games. `ar25` captures the *gist* (each directional action recolors all
   `9`-cells to an action-specific value → `cell_recall=0.914`) but gets details wrong
   (`heldout=0.0`), and is the single row with real dense progress (`hv_progress=0.333`).

The tell across patterns 1–4: the induced code models **static appearance or memorized cell edits**,
never the game's actual **action→effect dynamics**. Even the fully-correct exception (`sp80` t1,
`heldout=cell_recall=goal=1.0`) executed to `reached_levelup=False` with `plan_len=1` — the induced
win predicate fired one action in, which the real env did not treat as a level-up (an open question,
§6, likely the strict `is_level_complete` firing spuriously in-model or a `root_grid`/window mismatch
— not investigated further here).

---

## 5. Synthesis — the actual failure mode

**The induced world models are near-universally wrong about the game's real action→effect
DYNAMICS: most are identity, single-hardcoded-click, confabulated, or window-memorized engines
that reproduce some changed-cell *locations* (`cell_recall` 0.3–0.9) but never a full transition
(`heldout_accuracy = 0.0` in 29/37 rows) and are paired with degenerate goal predicates — the rare
near-perfect inductions occur only when the mechanic coincides with a game template the LLM already
knows (2048 = `sp80`).** `heldout_accuracy` is strict but *honestly* strict, not a metric artifact:
it correctly reports that these models never reproduce reality; `cell_recall` and
`goal_predicate_accuracy` are the softer, partly-misleading numbers (the latter inflated by no-op
class imbalance). The zero-level-up floor is therefore **downstream of the models simply being
wrong** — plan-search and candidate-selection are operating on a broken world-model substrate, so
even the ungated planner (and even a change-weighted `cell_recall` trust gate) cannot rescue plans
built on incorrect dynamics (0/11 plans reached a real level-up).

This is fully consistent with tonight's sibling findings: REQ-ARC-FCP-5757 (candidates
generated/ranked correctly 93 %+ → the selection layer is not the problem) and the search-architecture
audit (Carnot already has more search machinery than the winners). If dynamics induction is this
wrong, `plan_in_model`'s lookahead has *nothing correct to search over* even when invoked — the
binding wall is **generating a correct world model**, not selecting/searching among candidates.
(Converges with `project_arc_actions_to_progress_metric`: "bottleneck = dynamics-induction; heldout
uniformly 0.0.")

---

## 6. Open questions for a follow-up (NOT a fix plan)

Clearly separated from the diagnosis above; a future fix-building task decides what to target.

- **Why does even a perfect induction (`sp80` t1, `heldout=cell_recall=goal=1.0`) execute to zero
  real level-up with `plan_len=1`?** Candidate explanations not tested here: the strict
  `is_level_complete` fires spuriously in-model; `root_grid = full_traj[0].grid` is not the state the
  real-env execute phase resets to; or the window's single level-up needs a longer real trajectory.
  Worth isolating because it means the *execute/planning* stage may have its own bug independent of
  induction quality.
- **`cell_recall` vs a true `change_accuracy`.** The GAP-WM-TRUST-GATE fix needs full-transition
  `change_accuracy` + `n_changes_correct`, which these artifacts do not capture. Deciding on a gate
  change requires an experiment that logs the raw per-transition predictions (or extends
  `WorldModelVerifier` to emit `change_accuracy`/`n_changes_correct` per the GAP's candidate design),
  then re-scores — and, per standing discipline, an adversarial review before any live-gate change.
- **The two failure sub-populations differ in kind and may need different levers.** Qwen's deficit is
  *code emission* (34/40 overran the token budget without closing valid code); ThinkingCap's deficit,
  once it emits, is *dynamics correctness* on novel/click mechanics. A "bigger output budget / better
  code-completion" lever addresses the former; a "ground the induction in more/structured
  transition evidence, or bias toward mechanic-class priors" lever addresses the latter. The 2048/`sp80`
  observation (template-match → perfect) hints that a mechanic-class prior may be the higher-leverage
  direction, but this note does not test it.
