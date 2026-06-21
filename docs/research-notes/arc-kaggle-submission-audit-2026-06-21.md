# ARC-AGI-3 Kaggle submission audit — why v3=0.08 + the fix ladder (2026-06-21 outer-loop)

4-way parallel audit of the Kaggle competition path (model-load robustness, action-budget waste,
scoring-alignment/budget-allocation, kernel correctness). The Kaggle competition (the $25K 6/30
milestone) scores the GENERIC agent on a HIDDEN/OOD game set — banked replays of public games score
~0 there. Per-level score = `min(human_actions/agent_actions, 1)²`, averaged over ALL eval levels
(unsolved=0). v3 = **0.08**.

## Verdict: the 0.08 cause is NOT yet known — measure first

The v3 run is a COMPLETE scored run (gateway handshake + imports correct for the current image). But
**whether the Qwen3.5-9B-MTP generator tier actually engaged is unrecoverable from the logs**:
`main.py` printed only "...complete."; the env-var wiring lived inside `if server and gguf:` with no
else; the agent launches `llama-server` with `stderr=DEVNULL` and returns False silently on OOM
(`arc_executable_world_model.py:535,541`). 0.08 is consistent with BOTH "LLM never fired" (gated
behind a 24/80-transition stall AND can OOM at `n_ctx=16384`+MTP on a 16GB P100/T4 → silent CPU
degrade) AND "LLM fired but the explorer+induction is weak on OOD games." **Do not invest in LLM-tier
tuning until the next run self-reports engagement.**

## Fixed this session (kernel-side, low collision — committed 19f438619)

1. **`submission_kernel/main.py` — LLM-tier visibility + health probe.** Logs resolved server/gguf
   paths, a GGUF filename match (no stale-`.gguf` rglob risk), and a one-shot probe that spawns the
   generator with the agent's REAL args (incl MTP), stderr captured → prints `LLM GENERATOR HEALTHY` /
   `FAILED TO LOAD (likely OOM … consider CARNOT_ARC_MTP=0)` / `LLM TIER DISABLED`. Never crashes the
   agent. **Does not change the operator-frozen stack** — it measures it.
2. **`prep_daily_submission.py` — bundle-time import+build smoke.** ABORTs if the staged agent won't
   import/construct (a core file caught mid-edit by the daily rsync = 0 on every eval game; the prior
   guards only string-checked `MAX_ACTIONS` + the `.so` leak).

## Operator decision, evidence-gated

- **`CARNOT_ARC_MTP=0` for the 16GB eval** (drops the ~4GB MTP self-draft; 5.9GB Q4 + q8 KV @16384
  then fits comfortably). NOT applied autonomously — MTP is part of the 2026-06-19 frozen stack. The
  health probe above will show `FAILED TO LOAD … OOM` on the next run if MTP is the cliff; that's the
  evidence to authorize the flip. The hook already exists (`arc_competition_agent.py:1238` reads
  `CARNOT_ARC_MTP`), so it's a one-line kernel env set, no core edit.

## Biggest raw score lever — CONDUCTOR-LANE (core agent files, .421 actively editing; do NOT race-edit)

- **Global wall-clock budget allocator (critical).** No global allocator exists — only
  `MAX_ACTIONS=400`/game (`arc_competition_agent.py:1499`) + the kernel's 12h subprocess timeout. At
  ~262s avg wall/game over ~110 hidden games, a few slow induction-heavy early games can exhaust the
  8h cap and leave a large tail UNPLAYED (each scored 0) — a direct hit to the mean-over-all-levels
  score. Pass `remaining_wall/remaining_games` into `E3AgentPolicy.is_done` + the top of
  `_induce_and_plan`; force `is_done` when the per-game slice is spent; skip induction when the slice
  is gone.
- **Enable `early_stop_grace` (high).** `SUBMITTED_EARLY_STOP_GRACE=None` (`:72`) disables the
  early-stop that cuts the fruitless post-solve tail that "quadratically erodes the (human/agent)²
  efficiency score" (the code's own comment, `:889-893`). Set ~20–40; the consecutive-level-up reset
  (`:899-901`) keeps multi-level games uncapped.
- **Fix `is_done` phase-gating (medium).** `is_done` (`:1446-1447`) returns `explorer.is_done AND
  phase=='explore'`, so a level solved during `induce`/`execute` keeps stepping. Make target-reached
  terminate regardless of phase.
- **Cap LLM induction wall-cost (critical).** Induction fires on every level-up (`:1175`) AND every
  stall (`:1322`), up to `MAX_REFINEMENT_ROUNDS=3 × proposer.timeout=300s` = ~900s/episode, unbounded
  by the 10-steps/sec throttle. Add a per-game induction-episode cap (1–2) + lower the live proposer
  timeout to ~60–90s + only induct when wall remains.

## Research levers (generalization — the real ceiling on 0.08)

- **Cross-game scheduler:** `swarm.py` launches all ~110 games as concurrent daemon threads sharing
  one 16GB GPU; bound GPU-induction concurrency to 1–2 + fair per-game slices. A cheap shallow
  first-pass over ALL games (no induction) then a budget-bounded escalation only on games showing a
  level-up signal protects the index-weighted mean far more than per-game tuning.
- **No-progress watchdog:** abandon a game after ~60–100 actions of no new frame-hash + no level-up.
- **Per-game action cap proportional to the human baseline** once a level-up is seen (a 400-action
  "solve" of a ~15-human-action game scores ~0 under the squared term).
- **Faithful pre-submit gate:** add a mode to `arc_local_submission_gate.py` (kernel-territory) that
  runs the e3 policy WITH induction under a simulated global wall budget and reports
  `games_within_budget / projected_unplayed_tail` — the current gate disables induction, measures 8
  PUBLIC games, and models no cross-game wall-clock, so a global-budget regression is invisible to it.

## Next-submission runbook

1. Operator re-runs the daily prep (`scripts/kaggle/prep_daily_submission.py`) — now smoke-gated — and
   submits (operator-only). 2. Read the eval log for `LLM GENERATOR HEALTHY/FAILED/DISABLED`. 3. If
   FAILED-OOM → authorize `CARNOT_ARC_MTP=0` (one-line kernel env) and re-submit. 4. Route the
   wall-clock allocator + early-stop levers to the conductor (the real score movers).
