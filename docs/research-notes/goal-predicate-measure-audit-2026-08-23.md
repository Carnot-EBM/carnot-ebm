# Goal-predicate measure audit — 2026-08-23

## What was asked

`scripts/arc_induction_quality.py` scores emitted `world_model.py` files. It
classifies the goal predicate as `structural` or `flat`. Across the corpus it
reported 139 of 1245 models (11%) as `structural`. That number was about to be
read as the perception-grounding bottleneck, measured.

This note audits the measure before anyone uses the number.

## Answer in one line

**The 11% is wrong, and it is wrong in the direction that made it look like a
finding.** The true rate is **2.0%** (13 of 636 distinct models). The measure
over-reports because it substring-matches the predicate's raw source, and that
source includes comments and docstrings. **81% of the `structural` population
(59 of 73) hits a keyword only in prose, never in executable code.**

The bottleneck the number was reaching for is real, and is worse than 11%
suggested. But the original measure could not have established it.

## How the corrected number was reached

Four corrections, in order of how much each moved the number.

| Step | Population | structural | rate |
|---|---|---|---|
| As reported by the scorer | 1248 files | 139 | 11.1% |
| Dedupe identical file contents | 636 models | 73 | 11.5% |
| Require the keyword in CODE, not prose | 636 models | 14 | 2.2% |
| Also require the predicate can return True, and was emitted intact | 636 models | 13 | 2.0% |

### Step 1 — the file count is inflated, but the rate is not

1138 of 1248 files sit inside two git worktrees (`headwt`, `abpin`) of this
repo. Their `results/` trees are duplicate checkouts of committed fixtures. Only
636 of the 1248 files have distinct contents.

This corrects the denominator but **does not rescue the signal**: the rate moves
11.1% -> 11.5%. Duplication is not the bug. No file content spans two games, so
deduping is safe for per-game rates.

### Step 2 — the keyword list fires on comments (the real bug)

The scorer reads `ast.get_source_segment(src, goal_fn)` and asks whether any of
11 structure words appears in it. A source segment contains comments and the
docstring. Those are prose, not behaviour.

Split the predicate's tokens into code and prose. Docstrings and `#` comments
count as prose. Other string literals count as code, because `comp["object"]` is
real behaviour.

Result: **59 of 73 `structural` models (80.8%) have the keyword only in prose.**
The lost terms are `object` (44), `region` (13), `component` (2), `connected`
(2), `adjacen` (2).

The clearest example is `results/arc_e3/ar25/world_model.py`:

```python
def is_level_complete(grid):
    """
    The level is complete when the target object (color 4) perfectly overlaps
    ...
    """
    target_coords = np.argwhere(grid == 4)
    if len(target_coords) == 0:
        return False
    min_r = np.min(target_coords[:, 0])
    min_c = np.min(target_coords[:, 1])
    return min_r == 45 and min_c == 51
```

The word `object` is in the docstring. The code compares against the memorised
coordinates 45 and 51. This is a `flat` predicate that the measure scored
`structural`.

### Step 3 — a third of predicates can never report a win

Two defects the flat/structural axis does not see, over 636 distinct models:

- **27 (4.2%)** have no `return` statement at all. They always return `None`.
- **169 (26.6%)** return only the constant `False`. They can never say yes.

Union: **196 of 636 (30.8%) cannot report a win under any input.**

Separately, **57 (9.0%)** show a generator repetition loop or a truncated
emission. Example: `.../sk48__r2__on/sk48/world_model.py` is 379 lines. Its
`is_level_complete` starts at line 188, its whole body is `H, W = grid.shape`,
and the remaining ~190 lines are the same two comment lines repeated until the
output was cut off mid-word.

`structural` models are **more** likely to be dead than `flat` ones (42.5% vs
29.4%). A label that anti-correlates with basic health is not measuring
capability.

## The three checks that were asked for

**Does the matcher find the predicate? Yes — this is the one part that holds.**
Only 2 of 1248 files (1 distinct) score `ABSENT`. Zero of those have a win check
under another name. A generous regex for win-ish function names
(`won`, `solved`, `success`, `terminal`, `cleared`, and 15 more) found no
predicate the scorer missed. The interface is dictated by the induction prompt,
so the generator almost always emits exactly `is_level_complete`. **The 11%
denominator was not wrong for this reason.**

**Does it follow calls? It does not, and it almost never matters.** 559 of 562
`flat` predicates call no function defined in the same file. Only 3 of 562
(0.5%) delegate to a helper that contains a structure term. The blind spot is
real and tiny.

**Is the keyword list too narrow? Barely — the opposite problem dominates.**
Terms the list omits (`blob`, `cluster`, `segment`, `bbox`, `island`, and 11
more) appear in 246 of 562 `flat` predicates, but 245 of those are the single
word `shape`, and every inspected instance is `grid.shape` — the array dimension
attribute, not a derived shape. Reading the code confirms the `flat` label:
these predicates hardcode row indices, cell counts, and colour constants copied
from an observed win frame.

## An independent check that does not use keywords at all

Replace the vocabulary test with a behavioural one: does the predicate call a
primitive that derives structure from the grid (`argwhere`, `where`, `nonzero`,
`unique`, `bincount`, `label`)?

**32 of 635 (5.0%) do.** The two measures agree directionally. `structural`
predicates derive at 17.8% against 3.4% for `flat` — a 5x enrichment, so the
label is not noise. It carries real signal with a very high false-positive rate.

Even the 14 genuinely code-structural predicates are mostly memorising: 11 of 14
contain 5 or more distinct integer literals.

## Why the label inflates: it partly measures verbosity

| group | n | file lines | predicate lines | predicate branches | derives |
|---|---|---|---|---|---|
| true structural (code) | 14 | 169.7 | 37.8 | 22.0 | 28.6% |
| prose-only false positive | 59 | 109.0 | 10.8 | 2.5 | 15.3% |
| flat | 562 | 103.0 | 18.1 | 10.6 | 3.4% |

The false positives sit between the other two groups on every size measure. That
is the shape of a verbosity artefact: a longer predicate carries more comment
prose, so it is likelier to hit a keyword by accident.

## What the population looks like

Stated as correlation. The corpus is observational and cannot support a causal
claim.

- **31 games. 22 of them never produced a genuinely structural predicate.** The
  9 that did: ar25 (3), cn04 (2), lp85 (2), vc33 (2), tr87, sb26, sp80, re86,
  g50t (1 each).
- **Structural correlates with elaboration.** Structural models average 170
  lines, 4.5 functions, 86 branches, against 104 / 2.6 / 39 for the rest. N=14,
  so this is weak.
- **Today's live tu93 run: 0 of 3 genuinely structural.** This confirms the
  original observation at the corrected threshold, at N=3.

## What could NOT be determined

1. **No time trend.** File mtime is checkout time for the 1138 worktree files,
   so an mtime trend is an artefact. Run-directory date stamps give a real axis
   for only 496 of 635 models, and 139 sit in undated directories.
2. **The corpus mixes generator models.** `h2h_6440` ran Gemma-4-31B and
   Qwen3.8-27B; `baseline25` ran Qwen3.8-27B; the large July families predate
   the 2026-08-16 Qwen3.8-27B pin. A pooled rate across them is not a
   generator-quality rate for any one model. The recent small families show
   8-33% against 0-1% for the July families, but n is 3-26 and the generator
   differs, so this is not evidence of improvement.
3. **No link to level progress.** Nothing here joins a predicate's quality to
   whether the agent progressed. That needs a run that survives.
4. **The per-attempt population is empty.** REQ-ARC-WMTE-6690 archives every
   attempt, but only 2 attempt files exist so far (1 distinct). Every rate in
   this note is a survivor rate and carries the worst-attempt bias the scorer's
   own docstring names: re-induction fires on stagnation, so survivors skew
   toward models made under the worst conditions.

## What to change in the scorer

1. Strip comments and docstrings before the keyword test. This is the whole
   81% error.
2. Report the behavioural test (`derives`) beside the keyword test.
3. Report the two dead-predicate classes: no return, and only-returns-False.
4. Report the repetition-loop and truncation classes. At 9.0% they are a live
   generator-health problem and the current output hides them.
5. Deduplicate by content, and exclude paths inside a git worktree.

Item 1 alone changes the headline from 11% to 2%.

## Cross-references

- `scripts/arc_induction_quality.py` — the measure audited here. Its own
  docstring warns against exactly this failure ("a number that sounds like
  quality standing in for quality"); the warning was right and the keyword test
  fell into it anyway.
- Its cited exemplar, `greenrun/e3/ar25/world_model.py`, is genuinely
  structural: it calls `_label_components`, builds a goal mask, and tests
  adjacency. That claim in the docstring holds.
- CLAUDE.md "QA-Layer Authenticity Discipline" — the bug class is a pattern
  list narrower, and here also wider, than the concept it names.
- Memory `project_arc_live_agent_learning_gaps` — the perception-grounding
  hypothesis this measure was reaching for.
- Memory `field_names_lie_read_the_content`.
