# ARC scored-path held-out identity sweep, re-run on current code (2026-10-10)

Item 1 of `docs/research-notes/arc-heldout-test-scope-2026-10-10.md`. Same driver, same
pre-registered analysis, same 25 games, 2 arms, seeds 1 to 3, 400-action budget as the
2026-08-03 sweep. Only the code under test is newer.

Cells: `results/arc_heldout_identity_20261010/` (150 files, all status ok, 4,615 s wall with 6
workers, CPU only). Analysis: `results/outer_loop_arc_heldout_identity_rerun_20261010.json`.

## Result

| Measure | 2026-08-03 | 2026-10-10 |
|---|---|---|
| Games where the arms differ in banked levels | 0 of 25 | 0 of 25 |
| Control and held-out traces byte-identical | 72 of 75 | 72 of 75 |
| Seeds identical within an arm, per game | 25 of 25 | 25 of 25 |
| Banked levels, summed over 25 games (each arm) | 4.0 | 4.0 |
| Cells with zero actions or an error | 0 | 0 |
| Mean actions per cell | 389.1 and 389.2 | 389.4 and 389.4 |

- **Replicated:** removing the id-keyed knowledge changes nothing on the scored path. No p
  value is reachable because no game is discordant, so the null is bounded and not negative.
- **Changed underneath:** only 48 of 150 cell traces are byte-identical to August; 102 differ.
  The explorer behaves differently after two months of changes.
- **Two games swapped, both arms, every seed:** `cd82` went from 0 to 1 banked level and
  `tu93` went from 1 to 0. `sp80` (1) and `vc33` (2) are unchanged. The total stayed at 4.0.
  The new banking games are cd82, sp80 and vc33.

## What this does not show

- The scored path ran with the LLM-off stub proposer. The induction tier, the trajectory
  supervisor and the tool loop were not exercised. The two months of ARC work aimed at those
  parts are not measured here.
- Seeds are bit-identical, so three seeds are one replicate per game.
- The dev-twin numbers in the new analysis file (24 of 25 with the adapter, 10 of 25 without)
  are the August data read again. They were not re-run. The artifact says so in
  `rerun_note_2026_10_10`. Its `honest_verdict` text, copied by the analysis script, still
  describes the August dev-twin result.
- Why `tu93` lost its level and `cd82` gained one is not investigated. `tu93` is a game the
  registry records as deeply solved, so a loss is worth a look. It may be a real regression or
  a change in exploration order. I did not tell the two apart.

## A harness fault found and fixed on the way

The first launch produced 3-second cells with 0 actions and status ok. The game files
(`environment_files`) are untracked and exist only in the main checkout, and the code resolves
them relative to the repository root with no override. The worktree could not load any game.
The status field said ok anyway. I stopped the run, symlinked the directory in (removed
afterwards), and checked every new cell for actions greater than zero. The driver should treat
zero actions as a failure; it does not. That change is not made.

## Checks after the run

`git status` showed only the two new outputs. `results/arc_e3`, `ops/` and the August cells were
not modified. The default output path of the analysis script is the tracked August artifact, so
the new run passed `--out` explicitly.

## What this means

On the scored path, today's code behaves like August's: it banks almost nothing on public
games without the language model, and it does not depend on game identity. The informative
question is still item 2 of the scope note: does the induction tier install working plans on
the current stack. This run cannot say.
