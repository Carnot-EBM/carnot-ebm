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

## Correction and first induction run (2026-10-11, append-only)

**Correction.** Item 2 above says it "needs the GPU lease and the existing instrumentation, not new
code." That was wrong. The July driver (`results/first_win_llm_on_20260727/firstwin.py`) is bound
to the retired Qwen3.5-9B generator and to the main checkout path. A new driver was needed:
`scripts/experiments/outer_loop_arc_induction_plans_20261010.py`. It builds the shipped proposer,
pins one card by PCI bus order, and joins the server PID to the GPU UUID. Check at the start and
end of the run: confirmed on UUID `GPU-7971baff...` both times.

**Run.** 25 public games, variant 1, 4 cells in parallel, 200-action budget, one server. Wall time
29,905 s (8.3 h). Raw cells: `results/arc_induction_plans_20261010/cells/`.

| Count | Value |
|---|---|
| Induce attempts | 26 |
| Plan installed (planned) | 1 (tn36) |
| `proposer_failed` | 23 |
| `no_active_transitions` | 1 (vc33) |
| Games reaching level 1 | sp80, vc33 |

**This run does not give a rate.** In 23 of 26 attempts the proposer got zero responses. The
server log shows 94 `Context size has been exceeded` errors. The server ran 4 slots with a
unified KV cache of 106,496 tokens shared by all slots. Four concurrent induce calls, each near
the 62,490-token median, overflow it. This is my driver's choice (4 parallel cells). It is not
evidence about the model. The scored kernel plays one game at a time, so it does not hit this.

**What the run does show.** The one attempt with room (tn36) got a response and installed a
plan. July had 0 of 136 installed on the retired generator. One of one is not a rate. The
`FRAME-CHANGE SCORER LOAD FAILED` warning appeared in every cell, so the action-effect prior was
inert in this worktree. I have not checked whether the scored kernel sees the same failure.

**Next step if wanted.** Re-run with one cell at a time (`--k 1`). About 26 attempts at roughly
1,700 s each is about 12 hours on one card.

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

## Result of the single-cell rerun (appended 2026-10-13)

The rerun ran one cell at a time on card 1, with the shipped Qwen3.8-27B proposer
(`--k 1`). Commit `6e50c652f6`, directory `results/arc_induction_plans_20261011_k1/`.

| Measure | Value |
|---|---|
| Cells | 25, no cell errors |
| Induce attempts | 31 |
| Attempts that installed a plan | 10 (ar25, lp85, ls20, r11l, sc25, sk48, sp80, su15, tn36, vc33) |
| First wins | 5 (ar25, sp80, su15, vc33 reached level 1; lp85 reached level 2) |
| Main skip reason | heldout_transition_verification_failed, 13 |
| proposer_failed | 6, all generator timeouts |
| Wall time | 45.8 hours |

**Cause of proposer_failed.** The generator call timed out (`TimeoutError('timed out')`).
The server log has no error lines. The earlier context-overflow fault is not the cause here.

**What the 10 of 31 is not.** It is a count from one sample per game with no interval.
It is not a rate. Seven cells had a mostly or fully unavailable generator (bp35, cd82,
cn04, r11l, re86, tr87, wa30). The liveness lint flagged them, and they were added to
`ops/arc_llm_on_liveness_baseline.json` as timeout evidence. A plan rate on the other 18
cells is not computed yet.

**Comparison.** The July corpus recorded 0 of 136 installed plans with the retired 9B
generator. The two runs differ in generator, driver and cell layout, so the gap is a
direction, not a measured effect.

**Open.** Rerun the 7 timeout cells with a longer generator timeout (about 15 hours) to
fill the gap. Not started.

## Addendum 2026-10-17: the 7-cell rerun with a 7200 s induce timeout

The 7 cells with an unavailable generator ran again. Settings: one level-1 sample per
game, `--k 1`, `CARNOT_ARC_INDUCE_TIMEOUT=7200`, GPU 1 only (device confirmed by the PID
to UUID join). Results are in `results/arc_induction_plans_20261015_timeout7200/`. Wall
time was 23.6 hours (85,088 s).

| Cell | Generator calls / responses | Skip reason | First win |
|---|---|---|---|
| bp35 | 2 / 2 | heldout_transition_verification_failed | no |
| cd82 | 2 / 2 | heldout_transition_verification_failed | no |
| cn04 | 4 / 3 | heldout_transition_verification_failed | no |
| r11l | 4 / 3 | heldout_transition_verification_failed | no |
| re86 | 2 / 2 | heldout_transition_verification_failed | no |
| tr87 | 3 / 1 | proposer_failed (2 content failures) | no |
| wa30 | 2 / 2 | heldout_transition_verification_failed | no |

**What the rerun shows.**

- The timeout was the cause of the earlier generator failures. Generator errors fell from
  20 across the 7 cells to 0. The one remaining proposer_failed (tr87) is a content
  failure, not a timeout.
- No cell reached a first win, and no cell finished an attempt that passed the held-out
  check. Six of seven stopped at heldout_transition_verification_failed.
- No cell had a server failure. The generator was healthy after every cell.

**Correction to the first-run table above (append only; the table is not edited).** The
table says "Attempts that installed a plan: 10". That number is the count of the
`induction_attempts_planned` flag. The flag means the attempt reached the planning stage.
It does not mean a plan was installed. Two cells in this rerun show the flag set and
still skip with heldout_transition_verification_failed (cd82, r11l). The cleaner count is
attempts that were not skipped. In the first run that count is 5 of 31, and all 5 are
the 5 first wins. Treat "10 of 31 installed a plan" as an overstatement. Commit
6e50c652f6 carries the same wording in its title. I did not check how the flag is set in
the driver code, so the exact meaning of the flag is not confirmed.

**Plan rates, split as the earlier note asked.**

| Group | Cells | Attempts | Flag set | Not skipped | First wins |
|---|---|---|---|---|---|
| First run, 18 cells with a working generator | 18 | 24 | 9 | 5 | 5 |
| First run, 7 cells with a failing generator | 7 | 7 | 1 | 0 | 0 |
| Rerun, same 7 cells, 7200 s timeout | 7 | 7 | 2 | 0 | 0 |

These are counts from one sample per game. They carry no interval. The 7 cells were chosen
because the first run failed on them, so they are not a random sample. Do not compare
the groups as rates.

**Reading.** A working generator does not make the plan pass. On these 7 cells the
generator now answers and the induced model still fails the held-out transition check.
The earlier result that "6 of 31 attempts failed because of the generator" is closed.
The next limit is the quality of the induced model, not the generator timeout.

**Cost.** The longer timeout cost 23.6 hours of GPU 1 for 7 attempts and no wins. The
value 7200 s was my choice, not an operator decision.

## Addendum 2026-10-17 (second): why the 7 rerun cells failed the held-out gate

Read-only. Sources: the 7 cell files, `_induction_accept_threshold()` in
`arc_competition_agent.py:33`, and the accept test at `arc_llm_reinduction.py:1885`.

**The gate.** An attempt is accepted when `heldout_accuracy >= verifier_threshold`. The
live default threshold is 1.0, set by `CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD` when
present. Accuracy is an exact full-grid match. A code comment records that this gate was
met in 0 of 6 rounds on g50t in an earlier real run.

**What the 7 cells show.** The induced models are often close, not wrong.

| Cell | Held-out accuracy | Change fidelity | Spurious changed cells |
|---|---|---|---|
| bp35 | 0.92 (verify accuracy) | 0.998 | 0 |
| cd82 | 0.875 | 1.0 | 0 |
| wa30 | 0.852 | 1.0 | 0 |
| re86 | 0.630 | 0.992 | 9 |
| cn04 | 0.444 | 0.997 | 11 |
| tr87 | 0.88 (verify accuracy) | 0.907 | 7 |
| r11l | 1.0 (verify accuracy) | 1.0 | 0 |

Four cells (bp35, cd82, wa30, r11l) have change fidelity of 0.998 or higher and no
spurious cell. A threshold of 1.0 rejects them. I did not measure whether a lower
threshold would produce a plan that wins. That test is open.

**Not explained.** r11l shows perfect scores on the printed diagnostic and still skipped
with `heldout_transition_verification_failed`. The cell files do not store per-round
rows, so I cannot say which round or which check failed. The attempt archives under
`results/arc_e3/<game>/attempts/` hold the engines and may answer it. Not read.

**Limits.** Seven cells, one sample each, chosen because they failed earlier. The
diagnostic is the final round only. Two different metrics appear in the table
(`heldout_accuracy` and `verify_accuracy`), so the rows are not one scale.

## Addendum 2026-10-17 (third): matched llm_off arm on current code

Question: does the LLM induction tier add first wins over the same agent with the LLM off?
The July `llm_off` arm ran on July code, so no cell paired with the October run. This arm
runs on the same commit as the rerun (`62548c73e0`), variant 1, all 25 public games, K=4.
The driver gained `--arm llm_off`, which installs the no-op proposer and changes nothing
else. No server and no generator. Wall time 247 s. Results are in
`results/arc_induction_plans_20261017_llm_off/`.

| Measure | llm_off | llm_on (Qwen3.8-27B) |
|---|---|---|
| Cells, with cell errors | 25, 0 | 25, 0 |
| First wins | 1 (vc33) | 5 (ar25, lp85, sp80, su15, vc33) |

Paired by game: 1 cell won in both arms (vc33), 4 won only with the LLM on, 0 won only
with the LLM off. The four LLM-only wins are the four cells where an induced model passed
the gate. The 5 wins in the first run equal the 5 attempts that were not skipped.

**What this supports.** On these 25 games the LLM tier gave wins that the no-LLM agent did
not find. In July, with the retired 9B generator, the LLM arm won no more cells than the
no-LLM arm, so the generator change is a candidate cause. This run does not test that.

**What it does not support.**

- The sign test on 4 versus 0 discordant cells gives a two-sided p of 0.125. It is not
  significant at 0.05.
- One sample per game and one variant. The llm_off arm ran once, so run-to-run noise in
  the no-LLM agent is unmeasured.
- The `llm_on` column uses the 25-cell first run, with the 7 rerun cells swapped in. The
  swap changes no win count: neither run won on those 7 cells.
- This measures first wins on public games. It is a proxy for hidden-game performance,
  not that measurement.

**Next.** Variants 2 to 4 for both arms would give 100 paired cells. The llm_off side costs
minutes. The llm_on side costs about 45 hours of GPU 1 at the first run's pace.

## Addendum 2026-10-17 (fourth): the llm_off arm is deterministic

Two more llm_off runs on commit `62548c73e0`, CPU only: a repeat of variant 1, and variants
2 to 4 for all 25 games. Results are in `results/arc_induction_plans_20261017_llm_off_repeat1/`
and `results/arc_induction_plans_20261017_llm_off_v234/`. No cell errors.

- **Repeat of variant 1:** identical to the first run in win, reached level and action count
  on all 25 cells. The no-LLM agent shows no run-to-run noise here.
- **Variants 2 to 4:** 6 first wins in 75 cells (lp85~color02, sp80~color02, sp80~color03,
  vc33~color02, vc33~color03, vc33~color04). With variant 1's 1 win, the arm has 7 wins in 100.
- **Against July:** the July llm_off arm gave the same win or no win on all 100 cells. The
  agent code changed a great deal between July and now. None of it changed the no-LLM result
  on these cells.

**What this changes.** The 4-versus-0 comparison of the earlier addendum no longer has an
unmeasured no-LLM noise term. The baseline is fixed at 1 of 25 on variant 1 and 7 of 100
overall. The `llm_on` side is still one stochastic run per cell, so its noise is unmeasured.
The sign test result (p = 0.125) still stands.

**What it does not change.** The comparison still covers only variant 1 for `llm_on`.
Variants 2 to 4 with the 27B generator have not run. They cost about 45 hours of GPU 1 per
25 cells at the first run's pace. The six variant 2 to 4 cells that llm_off wins would
show whether the LLM tier keeps or loses wins the baseline already has.
