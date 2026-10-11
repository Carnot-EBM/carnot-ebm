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

## GPU lease findings (2026-10-10, after the reboot)

I looked at the lease code and at what actually protects a GPU run today. Nothing was launched.

**Both cards are back.** After the reboot `nvidia-smi` lists two idle RTX 3090s. The index order
changed: index 0 is now bus 03:00.0 (UUID `GPU-b52387a2...`), index 1 is bus 62:00.0 (UUID
`GPU-7971baff...`). The conductor still pins `CARNOT_ARC_GENERATOR_CUDA_GPU=0`, so it now
points at the other physical card than before the reboot. No llama-server is running.

**The GPU lease is not what protects a run.**

- `python/carnot/gpu_lease_phase_journal.py` is a sound design: one kernel `flock` per device UUID
  plus a checksummed JSON journal. It is used only by older experiment modules (6633, 6647,
  6764, 6899, 6986, 7013, 7079, 7086 and similar). `scripts/research_conductor.py` does not
  reference it, and neither does the live roadmap (0 matches).
- The default lease directory is `/tmp/carnot-gpu-leases` (override
  `CARNOT_GPU_LEASE_RUNTIME_DIR`). That is tmpfs, it does not exist now, and it is wiped by every
  reboot. A lease held in the default directory protects nothing across a reboot, and a lease
  nothing else checks protects nothing.
- What the real generator launcher does instead (`_cuda_gpu_has_headroom` and
  `_generator_cuda_min_free_mb` in `arc_executable_world_model.py`) is a free-VRAM check. It
  refuses to launch unless the card has the predicted need plus 1,500 MiB free. That is a race:
  two launches that start together can both pass it.

**So coordination today is convention plus that check.** The conductor uses index 0, the outer
loop uses index 1 (CLAUDE.md, 2026-06-27 allocation). I recommend not introducing a lease for
this run. It would add a mechanism nothing else honors.

**Launch recipe for item 2, from `memory/feedback_gguf_outer_loop_gpu_pinning.md` and the code.**

1. Pin one card, the one the conductor does not use. Set `CARNOT_ARC_GENERATOR_CUDA_GPU=1`.
   Do not copy the `"1,0"` default in `scripts/arc_holdout_generalization_probe.py`; that
   layer-splits across both cards and takes the conductor's.
2. Index and UUID can disagree between `nvidia-smi` and CUDA. After the model loads, join
   `nvidia-smi --query-compute-apps` by PID to the GPU UUID and confirm it is
   `GPU-7971baff...`. Record that join in the artifact.
3. Use a non-default port. The launcher reuses any healthy server on port 8919, including a slow
   iGPU one.
4. Set `CARNOT_LLAMA_SERVER` to the exact CUDA binary, and `CARNOT_ARC_SERVER_LOG_DIR` to the run
   directory. A mid-run relaunch can otherwise move to the iGPU with no visible sign.
5. Expect about 20,557 MiB predicted (20,352 MiB measured) at `n_ctx` 98,304 with 4 slots.
   Check `nvidia-smi` shows 24 GiB free on that card before the launch.

**Reapers that could kill the server.**

- `ExperimentTemplate.kill_gpu_zombies` killed servers until 2026-08-23. It now exempts
  llama-server and vLLM command lines (REQ-INFRA-079). I did not re-test this.
- `scripts/run_stop_authority.py` reaps an orphan llama-server only if all hold: parent PID 1,
  no systemd service cgroup, port referenced by no live process, no established connection, older
  than 2 hours, seen on two scans at least 25 minutes apart. A server whose launcher is alive and
  connected is safe. If the launcher dies, the server becomes reapable after 2 hours. Kill the
  server on exit.

**Residual risk.** A conductor task that loads a model with `llama_cpp` and `n_gpu_layers=-1`
may spread across both cards. If it starts while the measurement runs, one of the two fails
for memory. Check `nvidia-smi` before launch and record per-PID residency during the run.
Not measured: whether any current conductor task does this.

**Time.** About 1,730 s per median induction at 36 tok/s. Twenty attempts one after another
take roughly 10 hours. The conductor keeps running throughout, because it does not use card 1.
