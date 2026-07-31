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

> **CORRECTIONS 2026-07-31 (adversarial review). Three things about the number above, and one
> about the scope of what shipped. Original text left unedited per never-prune.**
>
> **W1 — p = 0.049 is the WITHIN-PROMPT figure; the primary inference is p = 0.125.** The 36 pairs
> are 6 games × 6 resamples of one induce prompt each, so the sign test's independence assumption
> does not hold for the population the claim generalizes to (new games). Clustered at the game
> level: **4 games better, 0 worse, 2 tied, p = 0.125**; whole-game sign-flip permutation on the
> net agrees exactly (+9, p = 0.1250). With 6 games the reachable floor is 2/2⁶ = 0.031, so
> significance required all six to improve. **What still carries the wiring decision** is that no
> game got worse and the cost effect is large enough to need no test at all — not a sub-0.05 p.
> See `arc-induce-repeat-penalty-confirm-2026-07-31.md` C2.
>
> **W2 — attribute the effect to the COMPOUND, not to `repeat_penalty`.** Penalty-alone is
> p = 0.1185 within-prompt / 0.625 clustered; the re-ask alone is p = 0.5. Neither component is
> individually significant at either level; only the compound reaches 0.049 / 0.125. The
> "penalty carries 11 of 13, the re-ask 2" split below is a decomposition of the point estimate,
> and remains the right reason to keep the two independently disableable — but it is not a
> significance claim about either part.
>
> **W3 — the numbers are conditioned on `n_ctx = 32768`, which is not the shipped default.**
> Phase 1 pinned `CARNOT_ARC_INDUCE_N_CTX=32768` because the shipped `_default_induce_n_ctx()`
> returns 81920, which does not fit a 24 GiB card and silently binds the iGPU. The cost half of
> the claim is completion-budget sensitive. This wiring shipped into that path without noting the
> gap; noted now. Until the 81920 default is fixed or the treatment re-measured at it, the
> validity and cost numbers hold at 32768 and are unproven at the live default.
>
> **W4 — SCOPE NARROWED, and this is a config drift from Phase 3.** As first wired, the penalty's
> condition was `codeonly_eligible` alone — which is also True for `_split_induce`'s focused
> goal-only call (`required=("is_level_complete",)`). Phase 1 sent ENGINE prompts only, so the
> penalty was reaching a prompt shape it was never measured on, while the source comment claimed
> the scope was "where it was measured" (true of the flag, not of the measurement). The defect
> gate was already correctly narrower. The penalty now uses the same condition, named
> `_engine_induce_call`. Narrowed rather than merely disclosed because with no measurement in
> either direction the goal-only call's correct state is the no-penalty baseline every banked
> result was produced under — and because `_split_induce` exists precisely to isolate a call that
> does NOT exhibit the repetition-loop failure the penalty treats ("valid in ~3.5s where the
> combined call fails"). **Cost, stated rather than buried:** the Phase 3 pre-flight exercised the
> WIDER configuration, so the shipped code no longer matches that artifact byte-for-byte. Phase 3
> was a REFUSAL, nothing downstream depends on it, and its finding (byte-identical action traces;
> `n_plans_found` 0 in every completed cell but one) is not a function of the goal-only sampler.
> Two tests pin the boundary from both sides; both mutations — widen back, and drop entirely —
> are killed.

## What is not claimed

That this fixes induction. The strict out-of-sample quality funnel moved 2/6 games → 3/6 on a
single attempt at p = 1.000, and that channel had only 5 discordant pairs — best reachable
two-sided p 0.0625, so it could not have shown an effect either way at that n. On tu93 and sc25 no
arm ever produced a correct engine, at any budget, in any of the 36 attempts. The bottleneck
Phase 1 pointed at is target-selection in click mechanics, and nothing here touches that.

> **W5 (2026-07-31) — sharper than the above: restricted to games where quality can be graded,
> it did not move at all.** On the three games whose held-out set contains a CHANGING transition
> (sc25, tn36, tu93), strict quality is **4 of 18 attempts in EVERY arm — identical**. The whole
> 2/6 → 3/6 movement is ft09, whose held-out set has zero changing rows and which passes only on
> the weaker zero-hallucinated-no-ops fallback. On tn36, the one game with a substantial gradable
> held-out set, both arms are 4/6 and each arm's best engine is the same 17/17 exact. **The
> treatment produced no better engine on any game where out-of-sample change prediction could
> actually be graded.** Also note (Phase 1 C7) that the quality channel was scored at induce
> **call 2**, while all three live gates and `_proposal_prefix` are on call 1 — so ft09's
> ungradability is an artifact of the scored call site, not a property of the game, and a quality
> claim would need re-running at call 1 regardless.

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

> **2026-07-31 (adversarial-review pass) — the same thing happened again, for the narrowing (W4),
> and it very nearly produced FALSE evidence.** The narrowing changes this file's hash, so the same
> 7 artifacts went stale again and the same 4 rebuilds were run. Same outcome — **0 substantive
> movement across 77,543 compared leaves**, the sole exception being 6012's
> `reproducibility_checksum`, which is a hash *over* the dependency set — so all 7 were again
> discharged by acknowledgement, `+8 −0` on every file, zero deletions.
>
> **The near-miss is worth more than the result.** A concurrent process reverted the working tree
> mid-sequence — the source file and both research notes were rolled back to HEAD while the git
> INDEX kept the correct staged content. Two of the four rebuilds therefore ran against the
> **pre-change** source and would have been written up as evidence for a change they never
> exercised. It was caught by reading each rebuild's own `provenance.code` sha rather than trusting
> that the file on disk was the file I edited. Every acknowledgement now cites that check
> explicitly. **The lesson generalizes: after a failed pre-commit run, verify the working tree still
> matches the index before trusting anything you then measure.** Recovery was `git checkout --` FROM
> THE INDEX (which held the good version), plus a copy of the source kept outside git so a
> concurrent checkout could not take it away again.

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

## Phase 3: the 12-cell grid is underpowered, and no GPU time was needed to know it

Phase 3 was gated on `python/carnot/analysis/treatment_activation_preflight.py`: run the pre-flight
first, and if the change does not perturb ≥ 6 of 12 cells attributably, do not run the grid.

**The pre-flight was not run live, because the grid it screens is refusable on arithmetic, and the
arithmetic is grounded in 36 paired live inductions rather than an argument.** The pre-flight's own
doctrine says the verdict is settled the moment the required count is unreachable, and that every
cell after that point is pure cost.

The bound is a strict one, and it does not depend on guessing a mechanism. **Banked levels is
downstream of the induced engine**: a cell where both arms produce the same induce outcome cannot
differ on banked levels. So the per-cell probability of a discordant *endpoint* is bounded above by
the per-cell probability of a discordant *induce* result — which Phase 1 measured directly, on the
real induce prompts of all six games.

`min_one_way_discordant_pairs(0.05) = 6`, so 6 of 12 cells must be discordant **and all six must
fall the same way**.

| perturbation channel | P(discordant)/cell | expected discordant of 12 | expected one-way | P(reach 6 one-way) | cells for a 50% chance |
|---|---|---|---|---|---|
| **A** — tracks strict quality (an engine good enough to pass the unchanged trust gate) | 5/36 = 0.139 | 1.7 | 1.0 | **0.000** | 68 |
| **B** — tracks mere usability (any mechanically-valid engine changes behaviour) | 17/36 = 0.472 | 5.7 | 4.3 | **0.238** | 16 |

**Channel B is the upper bound**, because `usable` discordance is the most permissive notion of
"the two arms differed at induce at all". So for the grid as specified:

> **P(the 12-cell banked-levels grid reaches p < 0.05) ≤ 0.24**, even granting that every induce
> difference propagates all the way to a banked level.

Roughly three runs in four would return an uninterpretable null *even if the treatment works
exactly as measured*. That is the same shape as the three nulls that paid for this pre-flight, and
it is why the grid must not be run at 12 cells.

**Which channel is real is the thing the live pre-flight would settle**, and it is worth settling,
because it changes the required grid from 16 cells to 68 — the difference between an affordable
experiment and one that is not worth running at all. The distinction is concrete: the defect gate
raises the supply of *valid* engines a lot (13/36 → 22/36) and the supply of *correct* ones not at
all (3 treatment-only against 2 control-only, p = 1.000). Whether the agent's action trace responds
to the first or only to the second is unmeasured, and the trust gate — which admits on correctness
and which the gate-rejection audit found was rejecting correctly every time — is untouched by this
change. Channel A is therefore the more likely of the two, and it is the one that says 68.

**What would decide it, and what it costs honestly.** A live A/B on 3–4 games with
`CARNOT_ARC_GENERATOR_SEED` pinned, both arms in one server process, plus an A/A control on the
same cells — the A/A is not optional, since a provably-same-code A/A diverged on 1 of 2 pairs on
2026-07-30, so residual nondeterminism survives the seed and its rate is unknown. On this path one
induction is 200–1200 s and an episode is many inductions, so this is hours, not minutes; a prior
probe died at 3.2 h with 4 of 12 cells never launched. It should be run against a wall budget with
the verdict checked as cells land, and it should report the per-cell perturbation rate — the
quantity that picks 16 vs 68 — rather than a go/no-go.

Nothing here is evidence that banked levels will move. Phase 1 says the tier now produces working
code more often, and says nothing about whether working code solves anything.
