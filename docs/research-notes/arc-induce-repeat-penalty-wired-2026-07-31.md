# The induce path now sends a repetition penalty and re-asks broken code — and the rebuild that was supposed to discharge it would have destroyed the record

**2026-07-31 · Phase 2 (wire) of the induce-reliability investigation**

Upstream: `docs/research-notes/arc-induce-repeat-penalty-confirm-2026-07-31.md` (Phase 1: 36 paired
attempts, 6 games, scored out-of-sample; CONFIRMED for validity and cost, NOT for quality).

## What shipped

Two changes to `LocalGGUFProposer` in `python/carnot/agentic/arc_executable_world_model.py`, both
scoped to the code-only induce path (`codeonly_eligible`), both env-reversible:

1. **`repeat_penalty 1.1` + `repeat_last_n 256`** on the `/completion` payload.
   `CARNOT_ARC_INDUCE_REPEAT_PENALTY=1.0` restores the previous payload byte-for-byte, because
   llama.cpp treats 1.0 as the identity.
2. **A defect gate + one plain re-ask.** A candidate that parses and defines the required
   functions is now dry-run and AST-checked through
   `arc_engine_static_validation.validate_engine_code`; on a mechanical defect it is rejected and
   re-asked once with a neutral block. `CARNOT_ARC_INDUCE_DEFECT_REASKS=0` disables it, so the two
   changes can be run independently — which matters, because the penalty carries 11 of the 13
   paired wins and the re-ask 2.

The re-ask text names **nothing** about the defect. That is the measured choice:
`repair_prompt_block()` exists and builds a defect-naming block, and head-to-head against the
neutral block it came back p = 1.000 over 5 discordant pairs. The defect *text* bought nothing;
the second *ask* is the whole effect. `repair_prompt_block` is deliberately left unwired.

**The safety invariant.** The gate can never convert an accept into a hard failure. It spends at
most one extra attempt, and only while an attempt remains (`attempt < tries - 1`); on the last try
the defective candidate is accepted exactly as before. I got this wrong in the first draft — the
`continue` fell out of the loop into the content-failure return — and the mutation check caught it,
not review. Removing the `attempt < tries - 1` guard is one of the three mutations killed by
`tests/python/test_arc_induce_repeat_penalty_and_reask_2026_07_31.py` (13 tests).

`arc_engine_static_validation` is now reachable from both live entrypoints on the ordinary import
closure, so its allow-list entry in `scripts/arc_orphan_solver_lint.py` was retired — verified by
temporarily unwiring the import and confirming the lint flags the module. The two tests that
pinned its orphan status inverted; their predecessors are described in their docstrings rather than
deleted, and both were written with this succession explicitly planned for.

## What is claimed

Validity and cost, on Phase 1's numbers: mechanically-usable engines 13/36 → 22/36 (paired sign
test p = 0.049, 17 discordant pairs), 47.2 s per attempt against 100.3 s, token-cap hits 20/36 →
2/36, `missing_return` 13 → 2.

## What is not claimed

That this fixes induction. The strict out-of-sample quality funnel moved 2/6 games → 3/6 on a
single attempt at p = 1.000, and that channel had only 5 discordant pairs — best reachable
two-sided p 0.0625, so it could not have shown an effect either way at that n. On tu93 and sc25 no
arm ever produced a correct engine, at any budget, in any of the 36 attempts. The bottleneck
Phase 1 pointed at is target-selection in click mechanics, and nothing here touches that.

## The rebuild would have destroyed the record — and the artifacts had already said so

Editing the module marks **7** registered artifacts stale. Four carry a `rebuild_command`. I ran
all four.

**Every one of them dropped history.** The rebuild regenerates each artifact from scratch, and the
generator does not re-emit post-hoc provenance, so it wiped:

| artifact | `freshness_acknowledgements` | `original_timings_superseded_by_rebuild` |
|---|---|---|
| experiment_6011 | 4 → 0 | present → gone |
| experiment_6012 | 4 → 0 | present → gone |
| experiment_6013 | 4 → 0 | present → gone |
| experiment_6021 | 2 → 0 | present → gone |

The preservation block those rebuilds deleted is itself the record of the *previous* time this
happened, and it carries the instruction:

> the same commit that caused this added VERIFIED-INERT acknowledgements to the artifact-freshness
> lint, which pin a new upstream hash plus a reason WITHOUT rerunning. **When the only drift is
> timing, acknowledge rather than rebuild.**

So the rebuilds were **discarded and the published artifacts restored** (`git status results/`
clean before re-annotating), and all 7 were discharged by acknowledgement instead. The diffs are
`+9 −0` on every file: not one existing line was touched.

This is a case where the instruction I was given ("rebuild in dependency order") and the
instruction the repository carries ("acknowledge rather than rebuild") disagreed, and the
repository was right — it had already paid for the lesson.

## The rebuilds were not wasted: they are the evidence

Having run them, the before/after pair is **differential execution at the artifact level**, which
is stronger evidence than an acknowledgement argued from reading the diff. Deep-diffed leaf by
leaf:

| artifact | leaves compared | substantive movement |
|---|---|---|
| experiment_6011 | 48,488 | **0** |
| experiment_6012 | 20,324 | 1 — `reproducibility_checksum`, a hash *over* the dependency set |
| experiment_6013 | 7,390 | **0** |
| experiment_6021 | 1,313 | 4 — `aggregation_wall_s` + 3 `code_provenance` fields |

**No published measurement moved.** 6021's head-to-head statistics (coverage 4/13 vs 12/13, meanB
0.062677 vs 0.384328, p_cov 0.007812, p_meanB 0.000977, N=13) are identical; 6013 is bit-identical
on every population count and decision arm.

The remaining 3 artifacts have no `rebuild_command` — they record completed live runs (one of
14.9 h) and cannot be recomputed from source. Their acknowledgements say so, and say explicitly
what is **not** claimed: that re-running them would reproduce their numbers. It would not, and this
change is one reason why. `outer_loop_arc_first_win_llm_on_eval_concurrency_20260727`'s own verdict
is `complete_UNFALSIFIABLE_treatment_not_applied_llm_output_never_reached_policy` — the exact
failure this wiring targets — so its re-run is a **new measurement**, not a refresh.

## A silently red test, found on the way

`tests/python/test_artifact_freshness_acknowledgement_2026_07_30.py` asserted that acknowledgement
evidence must "name where a reader can check it", implemented as
`"6011" in evidence or "changelog" in evidence`. Those two literals were a fair proxy when written.
They stopped being one as soon as an acknowledgement cited its evidence anywhere else: **4 of the 6
acknowledgements on each of the three capture artifacts already failed it at HEAD.** The test was
red and nothing surfaced it — it is not in any per-file hook's path.

Fixed by asserting the *property* (evidence cites a `results/`/`tests/`/`scripts/`/`docs/` path, an
experiment id, a commit sha, or the changelog) instead of two spellings of it. That is not a
loosened bar — a confident paragraph with nothing behind it still fails, which is mutation-checked.
It caught my own first draft, which cited `6011/6012/6013/6021` as bare numbers with no path.

The two genuinely location-free acknowledgements are dated 2026-07-30 and carry real evidence (one
is a 5,200-case differential execution) stated as prose. They are grandfathered **by date**, not
rewritten — editing historical evidence to satisfy a rule invented afterwards is what never-prune
forbids — and their count is pinned at exactly 2, so the exception cannot grow by back-dating.

## The budget single-source fix stays unlanded — measured, not assumed

`ops/pending-fixes/2026-07-31-induce-budget-single-source.patch` was to be landed here "if it does
not widen the artifact blast radius". It does, and the probe is exact:

* this change alone (`arc_executable_world_model.py`): **7 stale**
* plus the patch (`+ arc_competition_agent.py`): **12 stale**

The 5 it adds include `results/arc_per_level_reset_attribution_20260726.json`, whose rebuild is
`arc_per_level_reset_attribution_capture.py --budget 400` — a **live capture** that would replace
published numbers with fresh nondeterministic ones. Paying that to land a comment-and-constant
change whose value is unchanged at 4096 is not proportionate. The patch remains prepared and
applies cleanly.

## What the next measurement is

Phase 3 (the banked-levels grid) must not be run until
`python/carnot/analysis/treatment_activation_preflight.py` says the wired change perturbs ≥ 6/12
cells attributably. A refusal there is a success — it is the discipline three earlier nulls
lacked. Nothing in this note is evidence that banked levels will move; Phase 1 says the tier now
produces working code more often and says nothing about whether working code solves anything.
