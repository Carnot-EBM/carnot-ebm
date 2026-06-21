# ARC-AGI-3 live test-time-learning (TTT): scaffold + first offline gate (2026-06-21 outer-loop)

Operator reframe (2026-06-21): stop optimizing OFFLINE; LEARN each unseen game on the fly at live
runtime. Verified competition facts that force this: the agent gets only a frame→action INTERFACE (never
the game's code/rules), rate-limited to 10 REAL steps/sec (~288k steps / 8h over ~110 games); internal
compute (model fit, simulated rollouts) is FREE; score = `min(human/agent_actions,1)²` meaned over all
levels (unsolved=0). So the winning loop is: **gather a few rate-limited real transitions → learn a
transition model → plan for free inside it → commit only the short winning path**, with graceful give-up
that banks each reached level (the gateway records `levels_completed` continuously).

## Built this session (standalone, low-collision, offline-validated)

- `python/carnot/agentic/arc_live_ttt.py` — `LiveTTTWorldModel`: a per-game world model learned from
  played transitions, exposing the exact `engine(grid,action,data)->grid` + `is_level_complete` contract
  that the EXISTING `WorldModelVerifier` trust gate and `plan_in_model` BFS already consume. Layered:
  L0 exact transition table (full-bytes keyed, zero-train) + L1 `ObjectDeltaModel` (the existing zero-LLM
  rule learner). 5 unit tests green.
- `scripts/arc_ttt_validate.py` — the offline learning-curve gate: per game, collect transitions from the
  bundled `environment_files/` sim, split train/held-out, fit the learned model, and compare its held-out
  transition accuracy vs the frozen-9B-INDUCED engine. The de-risk BEFORE any submission.
- Key codebase finding: the learn-from-play substrate was ALREADY in the core — `E3AgentPolicy` fits
  `ObjectDeltaModel` every step (`_fit_dsl_model`) but uses it only for a diagnostic energy; the planner
  still calls the FAILING LLM engine (`e3.load_engine`, arc_competition_agent.py:1417). The real fix is
  to make the learned model the planning engine — which this module packages.

## First offline gate result — HONEST NEGATIVE (the gate did its job)

| game | learned held-out acc | LLM held-out acc | acc on NO-OPs | acc on CHANGING | held-out novel |
|---|---|---|---|---|---|
| cd82 | 0.20 | 0.00 | 8/8 (100%) | **0/32 (0%)** | 40/40 |
| ar25 | 0.25 | 0.525 | 10/10 (100%) | **0/30 (0%)** | 40/40 |
| ka59 | 0.20 | 0.20 | — | — | — |

**0/3 games clear the 0.5 trust gate.** The decisive diagnostic: the learned `ObjectDeltaModel` predicts
**100% of no-op transitions and 0% of state-CHANGING transitions** on entirely-novel held-out states. The
0.20-0.25 accuracy is *entirely* no-ops. So the rule hypothesis class (translate / object-translate /
recolor / click-recolor) **genuinely does not capture these games' change dynamics** — not a measurement
artifact. This is precisely the "biggest risk" the design named (the rule class can't express gravity /
growth / multi-object physics / timers / hidden state), materializing — and caught offline, before any
submission, exactly as the gate is meant to.

## What this redirects to — the neural learned-dynamics backend (operator-greenlit live-TRM)

The honest negative points directly at the operator's greenlight: replace the fixed-rule L1 with a
**learned neural dynamics model** (a small grid→grid CNN, or a TRM-style refiner) trained per-game on the
played transitions. Unlike a fixed rule class, a neural learner can fit arbitrary local change rules from
the ~120 transitions a probe gathers. This is build step 7 (`dynamics_backend='cnn'|'trm'`), now the
PRIMARY path rather than the optional fallback. The same harness re-runs unchanged to gate it: does the
neural model clear 0.5 held-out accuracy + beat the rule learner on the CHANGING transitions?

Open question the next gate answers: **can a tiny net learn these mechanics' exact full-grid transitions
from ~120 examples?** If yes → it becomes the planning engine (conductor wires it in per build step 8). If
no → the per-game-from-scratch-learning thesis has a sample-complexity wall on these mechanics, and the
honest fallback is the bare explorer + a richer-but-still-cheap structured learner.

## Build order status
1-2 (module L0+L1) ✓ · 3 (offline gate) ✓ · **7 (neural backend) — NEXT, now primary** · 4 (ValueNet
adapters), 5-6 (give-up watchdog + global scheduler) follow · 8 (conductor wires the winning engine into
`_induce_and_plan` + flips `SUBMITTED_EARLY_STOP_GRACE`) only AFTER the harness gate passes.
