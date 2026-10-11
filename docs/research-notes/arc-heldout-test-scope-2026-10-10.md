# ARC held-out test: scope (2026-10-10)

Operator question: "Have we made any real progress toward the ARC challenge?" and then "Yes" to
scoping a clean leave-one-game-out test. This note records what the record already shows, what
is still unknown, and the smallest next runs. Nothing was run to write it. Every number below
comes from files named in the text.

## Correction to my earlier answer

I said the only real test of transfer was the 2026-09-08 run and that I found no later result.
That was wrong. A pre-registered 150-cell measurement from 2026-08-03 exists. I missed it
because I read the memory index and the submission file, not the commit history. I also said a
clean test should block the registry entry, the solve file and the adapter. For the scored
path, that was the wrong target: the 08-03 pre-registration shows the adapter module is not in
the scored agent's import closure at all (55 files, `arc_game_adapters` absent).

## What is already measured

Source: commit `0bdc115fdf`, pre-registration
`docs/research-notes/arc-heldout-identity-prereg-2026-08-03.md`, cells in
`results/arc_heldout_identity_20260803/` (150 files, all status ok).

| Question | Result |
|---|---|
| Dev twin (`arc_loop_solve`), adapter on: games that bank level 1 | 24 of 25 |
| Dev twin, adapter free, shipped 6000-expansion budget | 10 of 25 |
| Dev twin, adapter free, 5 times the budget | 14 of 25; 11 games still resist |
| Sign test at game level (adapter on vs free) | 0 better, 11 tie, 14 worse; p = 0.000122 |
| On games it does solve, adapter-free move count vs adapter | the same. The deficit is search reach, not solution quality |
| Scored path (`E3AgentPolicy`, 400 actions): id-keyed knowledge removed | 0 of 25 games differ; 72 of 75 action traces byte-identical |
| Scored path banked levels, both arms (I summed the cell files today) | 4.0 levels summed over 25 games; only sp80, tu93, vc33 bank any |

The three seeds are bit-identical in both arms, so they are one effective replicate. The scored
path runs with a stub proposer that never generates (LLM off). The cell driver's docstring
(`outer_loop_arc_heldout_identity_cell_20260803.py`) records that an LLM-off arm matched its
LLM-on control in 74 of 74 cells, and that the LLM induction tier installed a plan in 0 of 136
LLM-on induce attempts. That was measured in August on the code of that day. I did not
re-check the 74 of 74 or the 136 figure against their source cells.

The 2026-09-08 run (`results/experiment_7144_v627_rebudgeted_arc_loo.json`) had the LLM on, one
game (r11l), control 1 level against withheld 0. It disqualified itself: the withheld arm still
read `ops/arc_solve_registry.yaml` and `results/arc_loop_solve_r11l.json`.

## What this means

- Generic search with no per-game adapter gets the first level on about 40 percent of public
  games (10 of 25), and about 56 percent with 5 times the search (14 of 25). That is a real
  measure of how far reusable method alone goes. It is not zero and it is not close to 25.
- On the scored path, removing id-keyed knowledge changes almost nothing. Either the knowledge
  never reaches the agent in a usable form, or the agent cannot use it. The data cannot say
  which.
- The scored explorer on its own banks very little (4 levels over 25 games in 400 actions).
  That fits the leaderboard scores of 0.02 to 0.12.
- The open question is not "does the adapter transfer". It is whether the LLM induction tier
  on today's stack installs plans that work, which the August data put at 0 of 136.

## What I recommend, smallest first

Do not write a new leave-one-game-out module. Four were written in one day in September, about
5,700 lines, for one number, and they produced 17 distinct defects. Reuse drivers that already
ran.

1. **Re-run the 08-03 scored-path sweep on current code.** Driver:
   `scripts/experiments/outer_loop_arc_heldout_identity_driver_20260803.py`. CPU only, no GPU.
   The August cells took 33,648 cell-seconds in total (mean 224 s). With 8 workers that is
   about 70 minutes. Question answered: has the scored path's dependence on id-keyed knowledge
   changed in two months of work (tool loop, supervisor, new kernel versions)? Gate fixed in
   advance: same sign test at game level. Expect 0 discordant games again; a different result
   is the finding.
2. **Measure induction on the current stack with the LLM on.** The metric is plans installed
   per induce attempt, over at least 20 games and the August denominator of 136 attempts. This
   needs the GPU lease and the existing instrumentation, not new code. Positive control: one
   game where the August run did install a plan, if any exists; if none exists, say so.
3. **Only if 2 shows plans installing,** repeat it with the held-out identity and an empty
   engine store, and deny the registry and prior solve files by path.

Cost: item 1 is cheap and safe. Item 2 needs both leased 3090s and the conductor must not hold
them. Item 3 waits on item 2.

## Not decided here

Whether to spend the GPU lease on item 2 while the conductor is running. The conductor owns GPU
0 and the outer loop owns GPU 1 (CLAUDE.md, 2026-06-27 allocation), so item 2 would fit on one
card, but that has not been checked against the current model size.

## GPU fit check for item 2 (2026-10-10, later)

Checked on the live machine; nothing was launched.

- **One 3090 is visible, not two.** `nvidia-smi` lists one card (bus 62:00.0, 24,576 MiB, 2 MiB
  used, no processes). The PCI bus shows a second RTX 3090 at 03:00.0 with no driver bound.
  Kernel and userspace driver versions match (615.71.09), so this is not a version split. The
  kernel log shows `nvidia-modeset` errors for `GPU:1` at 03:00 each day. I did not find the
  cause. The "outer loop owns GPU 1" allocation from CLAUDE.md therefore does not exist right
  now.
- **The conductor's generator targets that same card.** The service sets
  `CARNOT_ARC_GENERATOR_CUDA_GPU=0`, and the card is idle at the moment, with no llama-server
  running. A second generator would compete with the conductor's the day it starts one.
- **Memory fits one idle card.** Qwen3.8-27B Q4_K_M at `n_ctx` 98,304 with 4 slots: the envelope
  constants in `arc_executable_world_model.py` give about 20,557 MiB (my arithmetic from
  `_VRAM_QWEN38_*`); the measured point at that shape is 20,352 MiB. With the 1,500 MiB guard
  margin that is about 22,057 MiB of 24,576 MiB. It needs zero CPU offload on a free card. It
  does not fit beside another generator.
- **Time is the real limit.** The file records a median induction of 62,490 tokens, about
  1,730 s at a measured 36 tok/s on this card. August's denominator of 136 induce attempts
  would take roughly 65 hours one after another. 20 attempts would take roughly 10 hours.
  Parallel slots may shorten that; I did not measure it.

Consequence: a 136-attempt rerun is not practical now. A smaller run, on the order of 20
attempts, can answer one narrow question (do any plans install at all) but cannot give a rate
with a tight interval. It also needs the conductor to hold off its own generator, and a lease
check I did not look into.
