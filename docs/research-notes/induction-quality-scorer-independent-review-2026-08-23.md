# Independent review: the ARC induction-quality scorer and its reconciled rate

**Date:** 2026-08-23. **Author:** independent reviewer, working from the code and
the corpus only. The builder's notes and commit messages were deliberately not
read before the measurements below were taken, so these numbers are arrived at
rather than inherited.

**Scope.** `scripts/arc_induction_quality.py` — the two guards being added to it
(refuse a `results/` tree, refuse a nested repo clone, count distinct predicates)
and the reconciled "structural goal predicate" rate.

**Read-only.** Nothing under `results/**` was written. No GPU work was run.

---

## 1. What reproduces

Three numbers from the incident brief reproduce exactly under independent
re-measurement:

| Claim | Independent result |
|---|---|
| degeneracy 27.7% over the default root | 346/1248 = **27.7%** |
| degeneracy 26.6% deduplicated | 169/636 = **26.6%** |
| ~81% of "structural" hits are prose-only | 116/139 = **83.5%** (root), 59/73 = 80.8% (deduped) |
| executable-only structural rate ~2.0% | **1.8%** (root), 2.2% (deduped) |
| clone tree is ~91% of the corpus | 1138/1248 = **91.2%** |

One number does not reproduce, and the difference is a population definition, not
an error — see §3.

## 2. The branch-count concern is confirmed, and is worse than stated

The brief asked whether ranking by branch count rewards the worst memorisation.
It does, and the effect is measurable rather than anecdotal.

The four named games occupy ranks 1, 2, 4 and 13 of the deduplicated corpus by
branch count. Reading them settles what the count is counting:

- **ls20** (rank 13, 196 branches, 624 integer constants in 106 lines). Its
  `is_level_complete` is a row-by-row transcription of the win grid:
  `if grid[5, 4] != 15 or grid[5, 5:39] != 3 or ...`. It is also **not runnable**
  — comparing an array slice to a scalar and branching on the result raises
  `ValueError: The truth value of an array with more than one element is
  ambiguous` (verified directly against numpy).
- **cd82** (rank 1, 435 branches). Same transcription shape, and its predicate is
  **provably always False**: within one `and` chain it asserts
  `np.sum(r2 == 4) == 2` and `np.sum(r2 == 4) == 5`. The level can never be
  reported complete.

Generalising both tells across the deduplicated corpus (635 models with a goal
predicate):

| | count | share |
|---|---|---|
| goal predicate raises at runtime (slice used as a truth value) | 49 | 7.7% |
| goal predicate provably always False (contradictory `and` conjuncts) | 4 | 0.6% |
| either | 52 | **8.2%** |
| ...of which the shipped scorer flags `degenerate` | **0** | — |

And the correlation with the ranking is the wrong way round:

- of the **top 20** by branch count, **6/20 (30%)** have a broken goal predicate;
- of the **bottom 20**, **0/20 (0%)**.

So branch count is not merely a weak proxy for quality. On this corpus it is
**positively correlated with brokenness**, because the models that transcribe a
win grid generate one branch per transcribed row. The scorer's `degenerate` flag
catches `pass`, `return grid`, and `return True/False` — three shapes of doing
nothing — and misses the fourth and largest shape, which is doing something
elaborate that cannot execute or cannot ever fire.

## 3. The reconciled rate: population moves it more than either correction

Two corrections were applied together (require the keyword in executable code;
follow local calls and widen the vocabulary) on a population that also changed at
the same time. Separating the three shows the population is not a minor term.

Rate of "goal predicate is structural", by method and population:

| method | default root | deduped root | live only | live deduped |
|---|---|---|---|---|
| raw source + narrow vocab (as shipped) | 11.2% | 11.5% | 22.7% | 19.6% |
| executable only + narrow vocab | 1.8% | 2.2% | 17.3% | 13.0% |
| executable + follow-calls + narrow | 2.0% | 2.5% | 19.1% | 15.2% |
| raw source + wide vocab | 68.4% | 70.2% | 80.0% | 78.3% |
| executable + follow-calls + wide | 46.9% | 49.0% | 74.5% | **71.7%** |

Three things follow.

1. **Population alone nearly doubles the shipped number** (11.2% → 19.6%) with no
   methodology change at all. A before/after comparison that changes both at once
   attributes a population effect to a method.
2. **Following local calls is a ~2 point effect**, not a large one: 13.0% → 15.2%
   on the live-deduped population, 1.8% → 2.0% on the root. Whatever raises the
   number, it is not call-following.
3. **The vocabulary is a free parameter and it dominates everything else.**

### 3.1 One word carries the widening

Adding candidate terms one at a time to the narrow list, on live-deduped
(N = 92, baseline 15.2%):

| term | models it newly counts | as share of N |
|---|---|---|
| `shape` | 43 | +46.7 pts |
| `count` | 11 | +12.0 |
| `col` | 11 | +12.0 |
| `row` | 10 | +10.9 |
| `size` | 7 | +7.6 |
| `cell` | 3 | +3.3 |
| `width` | 1 | +1.1 |
| the other 16 candidates (`blob`, `cluster`, `contour`, `centroid`, `bbox`, `island`, `segment`, `patch`, `pixel`, `sprite`, `tile`, `group`, `coord`, `area`, `height`, `bounding`) | 0 | 0 |

`shape` alone takes the rate from 15.2% to 62.0%. **42 of those 43 hits are the
numpy attribute `.shape`** — the grid's dimensions, which essentially every grid
function reads. Exactly one is a bare `shape` identifier. Including `shape` in a
"perception vocabulary" therefore measures "this code calls `.shape`", not
whether the model perceives objects.

Note also that the sixteen terms a reader would most associate with object
perception — `blob`, `cluster`, `contour`, `centroid`, `bbox`, `island` — fire on
**nothing**. The widening does not find perception that the narrow list missed;
it finds generic array bookkeeping.

### 3.2 What this means for a reported point estimate

Every cell in the §3 table is defensible from some pairing of a stated population
and a hand-written word list. The defensible range on the live population runs
from about **13% to about 72%**, and the choice that moves it most is whether one
word denoting an array attribute is in the list. A point estimate with a ±5 band
is therefore narrower than the evidence supports, unless the vocabulary is frozen
in advance and justified term by term. The honest form of this result is a range
with the vocabulary printed alongside it.

## 4. Fail direction: three primitives, measured

| behaviour | measured result | consequence |
|---|---|---|
| `Path.rglob` into a **symlinked** directory | does **not** follow | a symlinked corpus is invisible to the sweep; also a legitimately symlinked run dir is silently skipped |
| `Path.rglob` into an **unreadable** directory (`chmod 000`) | **no exception**, silently returns fewer files | fail-**open**: the population silently shrinks and nothing says so |
| a git **worktree**'s `.git` | is a **file**, not a directory | `(d / ".git").is_dir()` misses every worktree |

The unreadable-directory case is the one that matters. A guard that decides
"is this a clone?" by reading the directory will, on a directory it cannot read,
conclude "not a clone" and sweep it — while the sweep itself has already silently
dropped files from it. Both halves fail toward reporting a clean, smaller number.

## MISSED INPUT

Directory layouts that ARE a duplicated corpus and that a `.git`-based clone
detector plus a `results/`-path exclusion do **not** catch.

**A. Real, already present in the corpus — copies with no clone and no `results/`
dir.** 18 of the 110 live files (**16.4%**) are byte-identical redundant copies,
in 9 groups, none of which involves a nested repo or a `results/` path:

```
x5  h2h_6440/e3_qwen38_27b/ar25/world_model.py
    h2h_full/e3_w1_qwen38_27b/ar25/world_model.py
    repair_widen/e3_w1/ar25/world_model.py
    repair_widen/goalab/e3_control/ar25/world_model.py
    repair_widen/goalab/e3_treatment/ar25/world_model.py
```

Two consequences. The same file is counted five times in the "live" population,
so the live rate is inflated by the same mechanism as the clone, at smaller
scale. And `goalab/e3_control/ar25` is **byte-identical to** `goalab/e3_treatment/ar25`
— an A/B whose two arms are the same file. Only the distinct-count half of the
guard addresses this; the path-exclusion half cannot see it.

**B. Real — the survivor and attempt populations overlap.**
`retention_run/e3/tu93/world_model.py` is byte-identical to its own archived
attempt `retention_run/e3/tu93/attempts/wm_20260823T213816_263035__6aeefdac7ad89dcc.py`.
The docstring presents `survivor` and `attempt` as two populations to compare;
here one file is a member of both. Any survivor-vs-attempt comparison is
partly a comparison of a file with itself.

**C. Constructed — a copy that excluded `.git`.**
`rsync -a --exclude=.git carnot/ tmp/snap/` or `cp -a results/ tmp/results.bak/`
produces a full duplicated corpus with no `.git` anywhere. This is not
hypothetical for this working directory: it already contains `ab.bak`,
`av.orig`, `av2.orig`, `av3.orig`, `av4.orig`, `aca_backup.py`, `amp_backup.py`,
`awm_fixed.bak` — the operator demonstrably makes `.bak` and `.orig` copies as a
habit. A `.bak` of a run directory is caught by neither guard.

**D. Constructed — a git worktree.** `.claude/worktrees/<name>/` is a layout this
project creates by design (the `EnterWorktree` tool). Its `.git` is a file, so
`is_dir()` misses it.

**E. Constructed — a clone nested deeper than the scan depth.**
`tmp/experiments/run7/carnot-checkout/results/...`. The two clones in this
working directory sit at depth 1, so a guard tuned against them may only check
shallow children and still pass.

## 5. What I could not determine

- **Whether the live population should be 29, 92, or 110 models.** The brief
  states 13.8% degeneracy on 29 distinct live models. I measure 10.9% on 110 live
  files and 12.0% on 92 distinct ones, and 13.8% is not reachable from either
  (4/29 = 13.79%, so the stated figure implies a 29-model population). The 29-model
  set is presumably a further restriction — one pin, or one date — but it is not
  derivable from the corpus layout alone, and the restriction rule is not recorded
  anywhere I could find. **This matters**: three defensible "live" populations give
  10.9%, 12.0% and 13.8%, and nothing in the artifact says which is meant.
- **Whether any of the 52 broken goal predicates ever ran.** Static analysis shows
  they raise or are unsatisfiable; whether the harness ever called them, and what
  it did when they raised, needs the run logs, not the source.

## 6. Cross-references

- `scripts/arc_induction_quality.py` — the scorer reviewed here
- `results/arc_object_perception_ab_20260728/` — the retired-generator evidence
  tree that is ~91% of the default-root corpus (EVIDENCE: read, never written)
- `openspec/capabilities/arc-world-model-trust-energy/spec.md` — REQ-ARC-WMTE-6690
  (per-attempt retention), REQ-ARC-WMTE-6700 (the guards)
- CLAUDE.md "QA-Layer Authenticity Discipline" — the pattern-list-narrower-than-
  its-concept bug class this review is an instance of
- CLAUDE.md "Field names lie — read the content" — branch count is the field name;
  §2 is what reading the content showed
