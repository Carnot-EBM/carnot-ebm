# Adversarial review: the "11% structural goal predicate" claim

Date: 2026-08-23. Author: independent adversarial reviewer (outer loop).
Method: read-only. No GPU work. `results/**` read, never written.

## The claim under test

> Only 11% of induced world models (139 of ~1245) have a goal predicate that
> references structure the model derived. The rest are "flat". The agent
> therefore cannot recognize winning, and this is the live agent's binding
> capability constraint.

The number comes from `scripts/arc_induction_quality.py` (uncommitted at the
time of review).

## Verdict: REFUTED

Two independent failures. The causal conclusion is wrong. The number is also
wrong by about 3x for the population the claim describes.

The rate itself reproduces exactly: 139 structural, 1107 flat, 2 absent, of
1248 files. So the arithmetic is not the problem. The population and the
inference are.

## 1. The scored agent does not use the predicate to detect a win

This is the decisive finding. Level recognition comes from the ENVIRONMENT.

The chain is short and unbroken:

- `arc_competition_agent.py:3264` sets `lvl = _level_of(latest)`.
- `arc_competition_agent.py:761-764` forwards to `_levels_completed(frame)`.
- `arc_agi3_live_adapter.py:352-353` reads the env field:

```python
def _levels_completed(frame: Any) -> int:
    return int(getattr(frame, "levels_completed", 0) or 0)
```

That value sets `self.best_level = max(self.best_level, lvl)`
(`arc_competition_agent.py:3280`). `best_level` is the only input to
`_current_goal_reached()` (`:5849-5852`). `best_level` also drives level-up
events, win-exemplar capture (`:5832`), and re-induction triggering. The
induced `is_level_complete` appears nowhere in that chain.

The predicate has two real uses, and both are SEARCH HEURISTICS:

1. `_install_goal_bias` (`:5980-6033`) orders the best-first frontier.
2. `plan_in_model` (`arc_executable_world_model.py:8526`) is the terminal test
   for a BFS run INSIDE the induced model, at zero real actions.

Two details sharpen this.

- `_install_goal_bias`'s own docstring (`:5983-5987`) states that the default
  binary path gives the frontier "a CLIFF -- zero signal at every non-terminal
  state". So a perfect structural predicate still adds no gradient unless
  `CARNOT_ARC_GRADED_GOAL_BIAS=1`. That graded path is dominated by Hamming
  distance to the win exemplar and uses the predicate only as a terminal
  anchor.
- The trust gate never scores the predicate. `WorldModelVerifier(transitions)
  .score(engine)` (`:4103`) scores the ENGINE. `is_level_complete` is carried
  as an unscored field on `WorldModelCandidate`
  (`arc_world_model_trust_energy.py:63`).

### Empirical confirmation

Cross `baseline25/rows.json` (25 games, live E3 runs) against each game's
emitted predicate class.

| goal predicate | games | leveled up | mean levels |
|---|---|---|---|
| structural | 6 | 3 (50%) | 0.83 |
| flat | 19 | 9 (47%) | 0.58 |

Fisher exact p = 1.000. Mann-Whitney on level counts p = 0.312.

Nine of the twelve level-ups in that run (75%) came from games the metric calls
flat. cd82 and r11l reached 2 levels each on a flat predicate. Three of the six
structural-predicate games (cn04, sk48, tr87) scored ZERO levels.

The proposed binding constraint has no measurable association with the outcome
it is said to bind.

## 2. Largest error in the measure: the population is 91% not-live

`DEFAULT_ROOT` is `$CLAUDE_JOB_DIR/tmp`. `find_models` does
`root.rglob("world_model.py")`. That directory holds TWO full clones of this
repo, `abpin/` and `headwt/`, 569 models each, byte-identical trees. So the
sweep ingested the repo's own committed `results/arc_object_perception_ab_
20260728/` evidence tree, twice.

- 1138 of 1248 files (91%) are that committed A/B tree, not live emissions.
- Its `preregistration.json:16` records
  `"generator": "Qwen3.5-9B-MTP GGUF"`. That generator was retired
  2026-07-28. The live pin is Qwen3.8-27B (2026-08-16).
- Distinct files: 636 of 1248. Distinct predicates: 504.

Restricting to real live-run directories moves the headline:

| population | n | structural | rate |
|---|---|---|---|
| as reported | 1248 | 139 | 11.1% |
| distinct predicates | 504 | 72 | 14.3% |
| live-run files only | 110 | 25 | 22.7% |
| live-run, distinct | 87 | 18 | 20.7% |

The scorer's docstring opens "Score the world models the live agent actually
emitted." It does not. Its SAMPLING BIAS WARNING anticipated survivor bias but
not this. The `population: survivor|attempt` line reported 1246 survivor and 2
attempt, which reads as reassurance rather than the alarm it should be.

## 3. The keyword list and the missing call-following both cost real accuracy

The scorer reads only the predicate's own body via `ast.get_source_segment`,
and matches 10 hand-written keywords. Follow local calls one to two levels and
widen the vocabulary to position-deriving operations (`argwhere`, `nonzero`,
`np.where`, `unique`, `count_nonzero`, and `_find*` / `_player*` / `_head*`
helpers):

| population | scorer | call-following + widened |
|---|---|---|
| all files | 11.2% | 16.5% |
| live-run files | 22.7% | 49.1% |
| live-run, distinct | 20.7% | 47.1% |

23 distinct live predicates flip flat to structural. Repo-wide, 13 of 67 flips
are pure helper-delegation; the rest are vocabulary.

I audited my own widened list adversarially. It is the same class of instrument
and deserves the same suspicion. On 17 sampled distinct live flips: 9 clearly
correct, 7 arguable (derived counts anchored to hardcoded coordinates), 1
clearly wrong (r11l, where `_cell` is a bounds-checked accessor, not
perception). Honest corrected live rate: about 35%, plus or minus 5 points,
against a reported 11%.

The attack does NOT hold on the high-branch tail. The models with the most
`goal_branches` (cd82 at 376, vc33 at 356, ls20, sp80) transcribe the win grid
row by row into literals. That is the worst memorization in the corpus. The
scorer calls them flat and is right. A branch-count proxy would have called
them excellent.

## MISSED INPUT

Real emitted predicates the scorer classifies `flat` that plainly consult
derived structure. All come from the current pinned generator or recent live
runs, so the misclassification bites hardest where it matters most.

`h2h_6440/e3_qwen38_27b/sk48/world_model.py` derives per-color hole state from
the bottom bar, locates the head, locates top targets, and checks a spatial
relation. It contains none of the ten keywords, and every derivation lives in
sibling helpers the scorer never reads.

```python
def is_level_complete(grid):
    ...
    info = _get_bottom_info(g)
    holes = {C for C, (_, _, hole) in info.items() if hole}
    if not all(C in holes for C in TARGET_COLORS):
        return False
    head = _find_head(g)
    if head is None:
        return False
    y, x = head
    tops = _find_top_targets(g)
    ...
    if not all(ty == y + 1 for ty, tx in tops.values()):
        return False
```

`baseline25/e3/sp80/world_model.py` builds a goal MASK and a player MASK,
locates the player, and checks 8-neighbour ADJACENCY. The scorer's list holds
both `mask` and `adjacen`. The code does both operations and writes neither
word.

```python
    goal = (g == 4) | (g == 6)
    player = (g == 9)
    py, px = np.where(player)
    for r, c in zip(py.tolist(), px.tolist()):
        r0 = max(0, int(r) - 1); r1 = min(H - 1, int(r) + 1)
        c0 = max(0, int(c) - 1); c1 = min(W - 1, int(c) + 1)
        if np.any(goal[r0:r1 + 1, c0:c1 + 1]):
            return True
```

Three more: `retention_run/e3/tu93/world_model.py` (3x3 sliding-window sprite
detection), `h2h_6440/e3_qwen38_27b/re86/world_model.py` (`np.argwhere`
localization then a relational body check), and
`repair_widen/goalab2/e3_control/ar25/world_model.py` (derives wall bounds,
then scans every offset for an L-template).

## Attacks that did not land

**The predicate matcher is fine.** Only 2 of 1248 models score `ABSENT`. The
induction path enforces the name: `arc_induction_tool_loop.py:451` checks
`if "def is_level_complete" not in code:`, and `arc_induction_tools.py:260`
does the same. The denominator is not wrong in the way suspected.

**Survivor bias is real but not sizeable here.** Only 2 archived attempts exist
on disk, so no attempt-versus-survivor comparison can be run. It is dominated
by the 91%-not-live defect, which is a larger and different problem than the
one the docstring anticipated.

## What I could not determine

- Whether the predicate matters for `plan_in_model` throughput. It IS the
  terminal test there. How often that path fires on the scored agent, and
  whether flat predicates cause plan failures, needs run diagnostics
  (`termination_reason`, `ttt_prior_engine_plan_diagnostics`) I did not have.
- Correctness of any predicate. Structural vocabulary is not accuracy. A
  memorized literal that matches the win grid can work. A structural predicate
  can be wrong. The scorer's docstring says this. The claim built on it forgot.
- The true live rate to better than plus or minus 5 points. Both keyword lists
  are hand-written and unvalidated against ground truth.
- Whether the null generalizes. N is 25 games in one run. p = 1.000 is a real
  absence of association, not proof of no effect. The run is underpowered for
  a small one.

## The honest restatement

About a third of goal predicates from the current generator reference derived
structure. The rest are memorized literals. This does NOT stop the agent
recognizing wins: the environment supplies `frame.levels_completed`, and the
agent levels up at the same rate in both classes. It likely degrades
model-internal planning, which is unmeasured.

## Step 6: check, not prose

Two conversions are worth making. Both are cheap. Neither fires on legitimate
work.

1. The scorer must REFUSE to sweep a `results/` tree or a nested repo clone.
   Counting committed evidence as live output is the defect that produced the
   headline.
2. The scorer must report DISTINCT-predicate counts, so a duplicated root
   cannot inflate N.

A third is prose only, because a check would fire on honest work: a metric
whose docstring says "what the live agent emitted" needs its root asserted, not
assumed.

## Cross-references

- `scripts/arc_induction_quality.py` -- the scorer under review
- `python/carnot/agentic/arc_competition_agent.py` -- `E3AgentPolicy`, the
  scored path
- `python/carnot/agentic/arc_agi3_live_adapter.py:352` -- the env level signal
- `python/carnot/agentic/arc_executable_world_model.py:8526` -- `plan_in_model`
- `python/carnot/agentic/arc_world_model_trust_energy.py:63` -- the unscored
  predicate field
- `results/arc_object_perception_ab_20260728/preregistration.json` -- the
  retired-generator provenance
- CLAUDE.md "ARC Live-Path Reachability Discipline" -- the two live entrypoints
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" -- the
  discipline this review applies
- CLAUDE.md "Field names lie -- read the content" (memory) -- the trap here

---

# ADDENDUM 2026-08-23: reconciling 2% against 35% (append-only)

A parallel review reported 2.0% after requiring the keyword to appear in
EXECUTABLE CODE rather than in a docstring or comment, having found 81% of
structural hits were prose-only. This review reported about 35% on live files
after following local calls and widening the vocabulary. Both corrections are
legitimate. Nobody had run them together. This addendum does.

## The answer

Applied together, on LIVE-run files only, DISTINCT goal predicates, n=87:

**23% to 39%, best estimate about 30%.**

Not 2%, and not 11%.

| correction | live files (n=110) | live distinct (n=87) |
|---|---|---|
| A scorer as shipped | 22.7% | 20.7% |
| B executable-code-only alone | 17.3% | 13.8% |
| C call-following + widened vocab alone | 49.1% | 47.1% |
| D1 BOTH, original vocabulary | 19.1% | 16.1% |
| D2 BOTH, widened vocabulary | 42.7% | 39.1% |
| lower bound (D1 or semantic detector) | -- | 23.0% |
| upper bound (D2 or semantic detector) | -- | 39.1% |

The same table over ALL files reproduces the other review's number exactly:
A 11.2%, B 1.8%, D1 2.0%. So the 2.0% figure is right, and it is a
WHOLE-CORPUS number. It is not a live rate.

## Why the two numbers differed: the prose defect is a property of the retired corpus

Of 116 prose-only hits, 110 are in the committed `results/` tree and 6 are in
live runs. So the prose-only share of structural hits is 83% over the whole
corpus and 24% over live runs. The Qwen3.5-9B models wrote long explanatory
docstrings naming components and masks; the current generator does that far
less. Correcting for prose on the polluted population removes almost every hit.
Correcting for prose on the live population removes six.

This means the two reviews were not disagreeing about the same population. Both
numbers are dominated by the population defect this note already documented.

## What drives the width of the range

Decomposed over the 87 live distinct predicates:

| driver | predicates moved |
|---|---|
| cost of the executable-only correction | 6 |
| gain from call-following alone | 2 |
| gain from the widened vocabulary | 20 |
| gain from the semantic detector beyond both | 0 |

**Vocabulary choice drives the width.** The executable-only fix and
call-following together move 8 predicates. The vocabulary moves 20. Any future
argument about this rate is an argument about which words count, not about
comments or delegation.

## The executable-only correction has its own false negatives

It is not a clean strip of false positives. Applied alone it deletes true
positives, because the keyword list is the shared weakness. The clearest case
is `abpin/results/arc_e3/m0r0/world_model.py`, a textbook iterative flood-fill
connected-component count:

```python
    visited = np.zeros(grid.shape, dtype=bool)
    num_blocks = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 10 and not visited[r, c]:
                num_blocks += 1
                stack = [(r, c)]
                ...
                while stack:
                    curr_r, curr_c = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    ...
    return num_blocks == 1 and p1_len >= 7 and p2_len >= 7
```

The words "connected" and "flood" appear only in its comment. The CODE uses
`visited`, `stack`, `num_blocks`, `dr, dc`. The shipped scorer calls it
structural for the wrong reason. Executable-only calls it flat, which is
wrong. My widened list also misses it. All three vocabularies fail on a
hand-rolled flood fill.

## A vocabulary-free detector, as a floor

To bound the range without arguing about words, I added three AST signals that
read behaviour rather than names: a literal collection holding at least three
of the eight unit coordinate offsets; a `while <name>:` loop with `.pop()` or
`.append()` on that same name; and a `where`/`argwhere`/`nonzero` result that
is then unpacked, indexed or iterated. It fires on 14.9% of live distinct
predicates, and 7 of its hits are ones executable-only calls flat.

It is a FLOOR, not a replacement. It recognises three traversal shapes and
nothing else, so it under-counts by construction. Its value is that no keyword
list can argue with it.

## Calibration of my own widened list

Sampled 17 distinct live flat-to-structural flips by hand: 9 clearly correct,
7 arguable (a derived count anchored to hardcoded coordinates), 1 clearly wrong
(r11l, where `_cell` is a bounds-checked accessor, not perception). That is
about 73% precision on the increment, which is how the 39% upper bound
discounts to about 30%.

## Guards shipped (REQ-ARC-WMTE-6700)

`scripts/arc_induction_quality.py` now classifies every swept path before
scoring it, and excludes two classes by default:

- `nested_repo_clone` -- any ancestor below the sweep root holds a `.git`. The
  check uses `.exists()`, not `.is_dir()`, because a worktree or submodule has
  `.git` as a FILE, and the two checkouts that caused the incident were exactly
  that shape.
- `committed_results_tree` -- any path part below the root is exactly
  `results`. Matched as a whole path part, so `my_results_run` is not caught.

Exclusions are named in the report, never silent. `--include-non-live` restores
the old population and says the rates are not live rates. An all-excluded sweep
exits non-zero rather than reporting on an empty population.

The report now carries three population sizes: total files, distinct files,
distinct goal predicates. A duplicated root inflates only the first.

The docstring now warns that branch count is size, not quality.

### Measured on the real corpus before shipping

| | files | degenerate | structural | distinct files | distinct predicates |
|---|---|---|---|---|---|
| guarded (default) | 112 | 12 | 25 | 93 | 87 |
| `--include-non-live` | 1251 | 346 | 139 | 637 | 504 |

1139 paths excluded, all `nested_repo_clone`: 1138 committed A/B models inside
the two clones, plus one synthetic git-worktree fixture in another agent's
scratch directory. Zero live-run models were excluded.

The `committed_results_tree` rule never fires on THIS root, because both
committed trees sit inside clones and the clone rule is checked first. It is
not decorative. Verified directly: sweeping with the clone itself as the root
(`--roots .../abpin`) classifies the same file `committed_results_tree`,
because the root's own `.git` is deliberately not treated as a nested clone.
The two rules cover different sweep shapes.

Mutation-proved: 8 mutations (drop the `.git` check; weaken it to `.is_dir()`;
drop the `results` check; weaken it to a substring; drop the fail-closed exit;
drop the distinct-count line; drop the exclusion reporting; drop the
`--include-non-live` warning) each turn the suite RED. Source restored
byte-identical. 13 tests pass. `test_suite_mutation_check` confirms the run
rewrote no tracked file.

## What this addendum still does not settle

The point estimate. 23% to 39% is an honest range and the width is a genuine
disagreement about what counts as structure, not measurement noise. Closing it
needs a labelled ground-truth set, which does not exist. Nothing here changes
the main finding: the rate, whatever it is, does not gate the agent's ability
to recognise a win, because the environment supplies that signal.
