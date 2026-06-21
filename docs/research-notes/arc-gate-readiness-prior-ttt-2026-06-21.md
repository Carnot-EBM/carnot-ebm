# Gate-readiness verdict: the prior+TTT improves the WRONG bottleneck (2026-06-21 outer-loop)

> ## CORRECTION (2026-06-21, later — a live-process investigation overturned the "goal-induction" framing)
>
> The verdict below said *goal-induction* is the bottleneck. **That was a CORPUS ARTIFACT, not a
> live-process property** — the exact trap CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent"
> point 3 warns about. A 3-agent investigation established:
>
> - **Win-DETECTION is NOT the gap.** The LIVE solver (`arc_competition_agent.StepwiseExplorer` +
>   `arc_solver_kit`) reads the win signal from the **env frame** (`frame.levels_completed` via
>   `arc_agi3_live_adapter._levels_completed` / `arc_solver_kit:415`) — never from the corpus. The live
>   agent already knows when a level completes (the env tells it) and already reproduces 52 levels. The
>   `n_win_states=0` / 0-4 null came purely from the **static corpus never containing a level-up
>   transition** (`lb=la=0`); the level signal IS in the raw HF parquet (`actions_by_level`,
>   `levels_completed`) and was dropped at the shard-BUILD stage (`arc_human_replay_corpus.py`), upstream
>   of capture. The offline probe also never triggers a live level-up under a random explorer budget.
> - **The REAL remaining gap is EXPLORATION-TO-FIRST-WIN** under the action-efficiency score —
>   credit-assignment / sub-goal discovery. On an unseen game the agent gets a sparse terminal
>   level-completed bit but must DISCOVER which action sequence causes it, from scratch, within ~5n
>   actions/level (score `min(h/a,1)²` punishes wasted exploration). `StepwiseExplorer` does **undirected**
>   salient-action BFS — no bias toward the latent win-mechanic — so on OOD games it never triggers the
>   first level-up or burns the budget getting there. **That is the 0.08 wall.**
> - **Highest-EV PROCESS lever (no weights, transfers by construction):** a **goal-biased online explorer**
>   — induce candidate win-predicates ONLINE from frame-delta structure (object/color-count reduction,
>   color-disappearance, coverage/toggle — the hypothesis classes already sketched in
>   `arc_agi3_goal_induction.py`, which is NOT wired into the live path), seeded with NO win example,
>   confirmed/pruned against the env `levels_completed` signal as observed, and used as the **priority key
>   for frontier expansion** (replan-on-divergence, not exact-match halt). Pure reusable scaffolding for
>   `arc_solver_kit`/registry, applied fresh to never-seen games.
> - **Cheap secondary:** rebuild the corpus shards from the full parquet to fix `lb/la` — lets the
>   prior+TTT plan toward OBSERVED wins for action-efficiency on REVISITS. Marginal; not the score-mover.
>
> **What still HOLDS from below:** NOT gate-ready; the prior+TTT (dynamics) + cell-recall gate do not
> address exploration-to-first-win, so they don't move 0.08. **What CHANGES:** the lever is a goal-biased
> online explorer, NOT "goal/win-condition induction" (the env already supplies win-detection).
> See `results/` workflow output + the live-process investigation (2026-06-21).


**Question (operator):** are we gate-ready to spend a scarce L4x4 submission shot? The gate (offline-first
discipline) = an offline result beating both the TRM baseline AND our best prior submitted run.

**Verdict: NO.** The chain below shows the prior+TTT path — this session's main thrust — cannot move the
0.08 hidden-game score, for a reason more fundamental than anything we'd fixed: **solving unseen games is
goal-induction-bound, and the prior+TTT improves dynamics prediction (the wrong bottleneck).**

## The two baselines

- **TRM baseline: 0 / non-existent.** No measured TRM-on-ARC-AGI-3 number exists (gate ledger
  exp3971 records `trm_arc_agi3_result: null`, `baseline_levels_solved: 0`); TRM was retired 2026-06-18.
  Trivially cleared (offline Carnot solves 26 levels vs 0). Moot half of the gate.
- **Best prior SUBMITTED run: Kaggle 0.08** (hidden/OOD games, kernel v3/v5, 2026-06-19). THE bar. The
  audit triangulated that 0.08 is a generalization/solve-rate wall — kernel v5 WITH 2.7x action-efficiency
  still scored 0.08. (The three.arcprize "33 levels" is a separate public banked-replay surface that scores
  ~0 on the Kaggle hidden leaderboard; not the bar.)

## The measurement chain (all offline, zero-quota)

| Step | Result | Meaning |
|---|---|---|
| 1. Prior pretrain (e30) | held-out cell-recall 0.314 → **0.5485** | the cross-game dynamics prior genuinely transfers |
| 2. Gate probe v1 (exact-match) | **0/5** fires on LOO games | the live gate (exact-full-grid ≥0.5) is unmeetable by a 64×64 CNN → path inert |
| 3. Gate probe v2 (cell-recall) | **4/5** fires (warm), prior adds **+2** (tn36 0.14→0.87, lp85 0.32→0.59) | re-metricing activates the path; the prior demonstrably helps |
| 4. **Solve test** | **0/4** (FULL *and* CNN-only) | **the real wall: `n_win_states = 0` — no goal to plan toward** |

## The decisive finding (step 4)

`plan_in_model` / `plan_and_execute` returned **"no plan to is_level_complete in model"** for all 4 firing
games — because the goal detector is EMPTY. Root cause confirmed in the data: **every transition in every
game of `data/arc_transition_corpus/*.npz` has `level_before = level_after = 0`** (lb/la arrays all zeros).
The capture/ingestion never recorded level progression, so there is no win-state to plan toward.

This is more fundamental than the gate-metric bug we fixed. Even a perfect dynamics model plus the
cell-recall gate cannot produce a solve without a goal. **The prior+TTT improves dynamics prediction; the
binding constraint for solving an unseen game is goal-induction (inferring the win-condition), which the CNN
dynamics do not address.** That is coherent with the audit's "0.08 = generalization wall."

## What actually moves 0.08 (where effort should go)

1. **Goal-induction for unseen games** — the binding constraint. Inferring a hidden game's win-condition
   from exploration (reward/level-progress signal, sub-goal discovery, or an LLM-induced goal predicate)
   is what would let ANY solver (TTT or otherwise) crack a level it hasn't beaten. This is the lever, not
   better dynamics.
2. **Fix the corpus level signal (data bug)** — the ingestion dropped `level_progress` (lb=la=0 everywhere).
   Re-ingesting with correct level tracking would at least let the TTT path plan toward OBSERVED wins (the
   efficiency story — fewer real actions to re-reach a known win), even though it does not solve the
   first-solve goal-induction problem. Secondary, but cheap and worth doing for any future TTT work.
3. **The cell-recall gate + prior+TTT are sound infrastructure** — keep them (committed: `trust_cell_recall`
   metric, default still `exact` in the live agent). They are necessary once goal-induction exists; they are
   not sufficient alone.

## Shot-spending recommendation (June-30, ~2 L4x4 shots left)

- **Do not spend a shot on the prior+TTT** — it solves 0 unseen games (no goal). A submission would
  re-score ~0.08.
- Hold the shots until an offline result actually beats 0.08, OR bank one safe locked-in 0.08-config L4
  submission for the milestone so we are not empty-handed.
- The highest-EV offline work toward 0.08 is **goal-induction**, not dynamics.

## Artifacts

- `results/arc_pretrain_prior.json` — dynamics prior transfer (0.5485)
- `results/arc_ttt_loo_gate_probe.json` — gate probe (exact 0/5 → cell-recall 4/5)
- `results/arc_ttt_solve_test.json` — solve test (0/4; n_win_states=0)
- `scripts/arc_ttt_loo_gate_probe.py`, `scripts/arc_ttt_solve_test.py` — re-runnable probes
