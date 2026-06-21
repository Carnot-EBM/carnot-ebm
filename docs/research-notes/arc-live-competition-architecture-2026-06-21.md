# ARC-AGI-3 live-runtime competition agent: architecture + dev-harness (2026-06-21 operator vision)

Operator directive (2026-06-21): the **rate limit is the binding bottleneck**. Build a system that runs
the *entire* play budget, makes useful use of *every* API call, learns each unseen game on the fly via
high-speed internal simulation, banks per-level success, gives up + moves on when a game's budget is
spent (banking saved budget from early wins), remembers attempts to improve future passes, and runs in a
**continuous local dev loop** (simulate the API ourselves) with instrumentation so the outer loop refines
each component toward the highest score.

## The budget (the bottleneck), quantified

- **10 real steps/sec** (every action AND reset is a rate-limited `/api/cmd` call). **Preview play cap =
  8h wall → ~288,000 real steps** total, across ~110 hidden games (~2,600/game even-split). The **12h is
  the Kaggle *notebook* cap** (includes model-load/overhead), NOT play time. The 2026 play cap is marked
  UNCONFIRMED in the requirements doc → **verify it**; the pacer is parameterized regardless.
- Real interaction (gather transitions + commit actions) is rate-limited; **internal compute (model fit,
  simulated rollouts, search) is FREE**. So the entire design is: spend few real steps to LEARN, simulate
  for free to PLAN, commit only the short winning path (score = `min(human/agent,1)²` rewards few
  committed actions; reaching more games raises the index-weighted mean).

## Component architecture (organizing the vision into composable engines)

| # | Engine | Responsibility | Status |
|---|---|---|---|
| 1 | **Local API simulator** | deterministic offline arcade over `environment_files/` = the dev substrate (the only offline signal; 25 public games proxy the hidden eval) | ✓ exists (`arc_solver_kit.offline_arcade`) |
| 2 | **Global pacer / scheduler** | allocate the ~288k step budget across games; per-game give-up; **bank saved budget** from early wins/abandons for later games | ✓ v1 (`scripts/arc_compete_sim.py`); NEXT: COLD no-progress watchdog, two-pass (shallow sweep → escalate), dual step+wall budget |
| 3 | **Per-game learned world model** | learn each game's transition model from played transitions (rate-limited probes) | ✓ scaffold (`arc_live_ttt.LiveTTTWorldModel`); rule-learner insufficient (0% on changing transitions) → **neural backend NEXT** |
| 4 | **Internal-simulation planner** | BFS/best-first to a win INSIDE the learned model (zero real steps), commit the short path; replan on real-vs-model divergence | ✓ exists (`plan_in_model`); wire to the learned engine |
| 5 | **Per-level feedback loop** | detect level success/failure; on failure restart the gather→learn→plan loop for the next attempt; weight what was tried | NEXT |
| 6 | **Cross-attempt memory** | remember previous attempts per game (+ across games); weight to improve future passes; new-game insight transfer | NEXT |
| 7 | **Meta-strategy** | higher-level: which solver algorithm to try next, when a game class needs a different approach | NEXT |
| 8 | **Instrumentation + dev loop** | per-game gap + budget-utilization + authoritative score; RUN → refine ONE component → RE-RUN, keep only if score strictly improves | ✓ v1 (`arc_compete_sim` emits all of it) |

**Parallelism**: within a game, real (rate-limited) interaction interleaves with FREE background compute —
while one real step is spent, the simulator runs high-speed rollouts to choose the next move and refit the
model; a meta-engine reasons across games. The offline sim has no rate limit (all fast), but the harness
counts steps as the live-budget currency so the dev-loop numbers transfer.

## Built this session (the foundation)

- `scripts/arc_compete_sim.py` — the **global-budget dev-harness**: runs the agent stack against the
  offline simulator under a parameterized total step budget, with dynamic per-game allocation
  (`remaining // remaining_games`, so early wins bank budget), per-game give-up, full per-game + global
  instrumentation (cap/used/saved/levels/efficiency/give-up), and the **authoritative** score
  (`EnvironmentScoreCalculator` via `arc_leaderboard_eval.run_game`) so local can't drift from the
  leaderboard. Validated end-to-end.
- **Baseline it establishes**: the bare explorer scores **0.0 (0 levels, ~98% budget burned)** on
  ar25/ka59/cd82/sp80 — the generalization wall, now a measurable dev-loop number to drive up.
- (earlier this session) `arc_live_ttt.LiveTTTWorldModel` + `scripts/arc_ttt_validate.py` — the
  learn-from-play engine + its offline gate (the rule learner is insufficient → neural backend next).

## Build roadmap (each step re-runs the harness; keep only if score strictly improves)

1. **Neural learned-dynamics backend** for engine #3 (small grid→grid CNN / TRM, operator-greenlit) — the
   make-or-break test of learn-from-play; gate via `arc_ttt_validate.py`.
2. **Wire the learned engine into the planner** (#4) inside `arc_compete_sim` (a pluggable policy), measure
   the score lift vs the 0.0 explorer baseline.
3. **COLD give-up watchdog + two-pass scheduler** (#2) — stop starving the tail; bank-and-move-on.
4. **Per-level feedback + cross-attempt memory** (#5, #6) — restart-on-failure with weighting.
5. **Meta-strategy** (#7) — algorithm selection per game class.
6. **Conductor wiring** of the winning stack into the submitted agent (`_induce_and_plan` engine swap +
   `SUBMITTED_EARLY_STOP_GRACE`) ONLY after the harness shows a score lift; gated by
   `test_arc_submitted_agent_parity.py` (the 0.08-incident guard).

Verify before scaling: the 2026 play-budget cap (8h vs 12h) and whether a tiny net learns these mechanics
from ~120 examples (engine #3's open risk).
