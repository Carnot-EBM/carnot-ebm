# ARC-AGI-3 Focused Loop + the Engine Question (2026-06-17)

How we keep making rapid, measured progress up the ARC-AGI-3 leaderboard — and an
important architecture correction about what model the agent actually runs.

## What engine are we using? (the honest answer)

| Component | Engine | Notes |
|---|---|---|
| Competition explorer (`StepwiseExplorer`) | **NONE — training-free graph search** | No TRM, no LLM, no foundation model. Salience candidates + RESET-replay nav. |
| E3 world-model inducer (dev) | **codex / gpt-5.5 (CLOSED, ONLINE)** | Validated the loop, but **illegal in the offline eval** + violates decentralization. DEV-ONLY now (`offline_legal=False`). |
| E3 world-model inducer (competition) | **open local GGUF (gemma-4) via llama.cpp** | `LocalGGUFProposer`, OFFLINE, decentralized. The competition-legal default. |
| TRM (Sudoku-extreme ckpt) | **NOT on the ARC-AGI-3 path** | TRM is a static grid-puzzle refiner; ARC-AGI-3 is interactive. Wiring a TRM-class model would mean TRAINING one on game dynamics (an offline-legal learned engine — a real option, not done). |

**So: we are NOT using TRM here, and the core explorer uses NO foundation model.** The
only model in the stack is E3's proposer, and the competition-legal one is an OPEN LOCAL
model — never a closed online API. This is forced by TWO constraints that happen to
agree: (1) the competition eval has **no internet** (codex can't run), and (2) CLAUDE.md
decentralization rules 1–2 (local-first open models; closed integration optional, never
required). The Carnot VERIFIER grounds whatever the proposer writes, so a weaker local
model just earns a lower verifier score — honestly.

### Two offline-legal engine paths (both keep the engine LOCAL)
1. **Local-LLM E3** (built): `LocalGGUFProposer` induces a Python world model in-sandbox;
   the verifier grounds it; plan in the model. Quality of small-model induction is the
   open milestone the loop measures next.
2. **Trained TRM-class dynamics/policy** (option, not built): train a small recurrent
   model offline on ARC-AGI-3 transitions to predict next-frame / good-action; ships with
   the agent, runs offline. This is the natural "use TRM not a foundation model" path —
   it just needs a training corpus of game transitions (which our explorer already
   generates).

## The focused loop (the engine for rapid progress)

`scripts/arc_leaderboard_eval.py` is the measurement core. Each iteration is ONE small,
attributable, gated change:

1. **RUN** `arc_leaderboard_eval.py` — scores the agent FROM SCRATCH on the public games
   with the LEADERBOARD METRIC (levels + efficiency = `min(human/agent,1)^2`), zero
   quota, and writes a per-game **gap log** (which games stall + the failure signature).
2. **READ the worst gap** — a game stuck at L0, or a solved game with terrible efficiency.
3. **IMPROVE ONE ingredient** (single attributable change):
   - exploration: visual-salience PRIORITY TIERS, frontier-DISTANCE navigation (cheaper
     than RESET-replay), status/HUD-masking (E1 has the masking; tiers + frontier-dist
     are the next explorer upgrades, straight from arXiv:2512.24156).
   - depth: E3 world-model induction (local proposer) for a deep game where blind
     exploration can't reach the win in budget.
4. **RE-RUN**; keep the change only if total levels or efficiency strictly improved AND
   no previously-solved game regressed (the regression gate).
5. **LOG** the closed/!closed gap (never-prune) and repeat.

Baseline (2026-06-17): **2 levels from scratch** (lp85 at 11 actions = eff 1.0; sp80),
**9 open gaps** (worst: r11l, ls20, wa30, cd82, su15, tu93). Every future ARC change is
measured against this — progress-not-churn (north-star §1).

## Why this converges on the leaderboard

Scoring rewards BOTH solve-rate (levels) AND efficiency (squared human-action ratio).
The two Carnot levers map exactly: the graph-explore SOTA ingredients lift solve-rate on
shallow/medium games cheaply; E3 world-model planning lifts BOTH on deep games (plan in
the model = few real actions). The verifier is the through-line — it grounds the induced
model and prunes the search. The loop turns "make progress" into a measured, gated cadence
instead of churn.

Cross-refs: `python/carnot/agentic/arc_competition_agent.py` (StepwiseExplorer +
E3AgentPolicy), `python/carnot/agentic/arc_executable_world_model.py` (pluggable
proposer), `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`
(eval is hidden + offline), `docs/research-notes/arc-agi3-sota-ingestion-2026-06-17.md`
(E1/E3 levers).
