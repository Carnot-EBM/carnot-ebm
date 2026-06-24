# Leader-gap (StochasticGoose 1.21 vs our 0.08) weighed against the multi-level wall — forward path

**Date:** 2026-06-24 · **Author:** outer-loop (operator-directed) · **Deadline:** 2026-06-30 (6 days)
**Workflow:** `arc-leader-gap-online-action-learning` (`wf_dd932102-24e` / `wc2xzu09u`, 3 agents, verdict CONFIRMED)
**Verified against code** (not just the agents' report): the two linchpin claims below were re-read by hand.

This note answers the operator's explicit ask: *"When the ranked design + prototype the top lever lands,
weigh it against this multi-level wall and identify the forward path to explore to solve this challenge."*

---

## 1. The confirmed gap (hand-verified)

**StochasticGoose (leader, ~1.21)** — `arc-sota-refs/ARC3-solution/custom_agents/action.py`:

- TRAINS an action-effect CNN **online, per game, from its own experience**. No `torch.load`, no checkpoint,
  no pretrained weights anywhere — `action_model=None` at init (`:145`), constructed fresh from random init.
- **Self-supervised free labels:** for each played transition, `reward = 1.0 if frame changed else 0.0`
  (`np.array_equal(prev, cur)`, `:386-403`). md5 hash-dedup on a 200k deque for sample efficiency.
- **Cadence:** one Adam (lr=1e-4) BCE gradient step every 5 actions, batch 64 (`:255-298`, `:440`).
- **Coordinate head:** a per-pixel 64×64 click heatmap; clicks are **sampled hierarchically from the CNN**
  (action-type then coordinate, `:171-228`) — the net *proposes* the coordinate.
- **Dynamic reset on level-up:** on any score increase, buffer cleared AND model+optimizer re-initialized
  from scratch (`:334-361`) so it re-learns the *next* level's dynamics from zero.
- **No goal predicate, no planner.** It never induces "what is the win condition." It explores toward
  frame-change and lets the environment's score signal drive per-level reset.

**Ours (`E3AgentPolicy`)** — `python/carnot/agentic/arc_competition_agent.py` + `arc_frame_change_predictor.py`:

- Action-effect CNN is a **FROZEN cross-game prior** (`results/experiment_4629_live_frame_change_cnn.pt`),
  run `eval()`/`no_grad()` only — no optimizer, no `.backward()`, no per-game update. It is a **re-ranker**
  of a structurally-enumerated candidate set (object centroids from connected components), not a proposer.
- Coordinates come from object centroids (`rich_action_candidates`), NOT a CNN coord head; the CNN heatmap
  only *looks up* a score at each pre-baked centroid.
- We DO have an online CNN (`arc_live_ttt.CNNDynamics`, real Adam/CE `fit`, warm-started from
  `models/arc_dynamics_prior.pt`) — but it is **gated OUT** on every hidden game (verified: `arc_live_ttt.py:382`
  default `trust_metric="exact"`; `:407` `gate_value = acc`; the call site `arc_competition_agent.py:2049`
  passes the default). The gate demands EXACT-full-grid match, which reads ~0 for a 55%-cell-accurate CNN —
  its own docstring (`:315-320`) says so. A granularity-matched `trust_cell_recall` already exists but isn't wired.
- Even when that engine *does* pass, it feeds `plan_in_model(_eng, _isdone, ...)` (`:2053`) — i.e. it still
  plans toward the induced **goal predicate**, so it inherits the goal-grounding wall.

**Verdict: CONFIRMED.** The core gap is **online action-learning**: they adapt to the hidden game; we don't.

---

## 2. Why this matters for BOTH problems at once

There are two problems, and the leader analysis shows they have one shared root and one shared fix.

### 2a. Why the leader deepens multi-level and we don't (the architectural insight)

Our multi-level wall (`docs/research-notes/multi-level-deepening-diagnostic-2026-06-23.md`,
known-issues `.430`) is the **degenerate L2 goal predicate**: at level-up the active-transition window has
zero L2-win positives, so the LLM induces `is_level_complete` for L2 from nothing, it is never verified, and
`plan_in_model` returns `no_reachable_plan`. This wall is a **self-inflicted consequence of our
goal-induction-first architecture** (explore → induce a GOAL → plan toward it → execute).

**The leader has no goal predicate and no planner, so it has no goal-grounding wall.** Its multi-level
deepening is *free*: level-up → reset → re-learn the next level's dynamics → keep exploring toward
frame-change. It never needs to know what the win condition *is*; the environment's score tells it. The #1
leaderboard agent proves the multi-level wall is **not fundamental** — it is an artifact of requiring
goal-induction before action.

### 2b. Why the leader's online learning works where ours is gated out (easy vs hard target)

- Leader learns an **EASY** signal: *binary* "did something change?" — free labels, near-perfectly
  learnable, drives action+coord selection directly.
- Ours tries to learn a **HARD** signal: the *exact* next 64×64 grid. 55% cell-accurate → exact-grid trust
  ~0 → **we gate ourselves out** for failing a target we never needed.

We picked too hard a learning target and then disqualified ourselves on it. The leader picked the right
target and it's the whole reason their online loop functions.

---

## 3. Weighing the top lever against the wall (the honest verdict)

The workflow's ranked levers:

| # | Lever | First-win impact | Effort | Crosses the multi-level wall? |
|---|---|---|---|---|
| 1 | Online action-effect **re-ranker** (`OnlineActionEffectScorer`, Adam BCE every 5 actions, per-level warm-reload) | HIGH | 2–3d | **No — by itself.** As designed it re-ranks object-centroid candidates feeding the goal-induction path; deepening still routes through `plan_in_model(... _isdone ...)`. Necessary, not sufficient. |
| 2 | Fix gating to `trust_cell_recall` (un-gate our existing online CNN) | MED | <1d | **No.** Un-gates the world-model engine but it still plans-toward-goal. Cheap first-win floor only. |
| 3 | CNN **coord head proposes clicks** (top-k heatmap pixels as ACTION6) | HIGH ceiling | RISKY 3–4d | **Toward yes** — this is the leader's actual driver. Combined with online training + reward-driven reset it becomes goal-FREE deepening. |

**Key finding for the operator:** the workflow's #1 lever (online *re-ranker*) helps first-win (the deadline
metric) but **does not cross the multi-level wall on its own**, because we keep the goal-induction planner as
the *driver* and demote the CNN to a re-ranker. The lever that crosses the wall is the **fusion of #1 + #3**:
the online frame-change CNN as the **driver** of action+coordinate selection, reward-driven, with per-level
reset — i.e. *adopt the leader's loop*, not just its CNN.

---

## 4. The forward path (converges with the north star)

The leader-gap analysis and the multi-level-wall diagnosis converge on the **same** prescription, and it is
the project's own north star (`project_arc_agi3_north_star`: *"generator induces, verifier
routes/prunes/verifies — NOT induces"*). Today we have it **inverted**: goal-induction DRIVES (plan-toward-goal)
and the frozen CNN merely re-ranks.

**Flip the architecture:**

1. **Promote a goal-FREE online-exploration policy to the live agent's primary deepening loop** (the
   leader's loop): an online-trained *binary frame-change* CNN (free labels, the EASY target) with a
   **coordinate head that proposes clicks directly**, driven by the environment's reward (score increase →
   per-level reset), NOT by an induced goal predicate. This structurally sidesteps the goal-grounding wall AND
   adapts per-hidden-game — helping first-win (deadline) AND multi-level (the wall) **simultaneously**.

2. **Keep the energy verifier / goal-induction as the SECONDARY router/pruner, not the driver.** This is
   exactly the north star and the oracle-distinct moat: the online CNN is the *generator* of action
   proposals; the energy verifier *prunes/routes* among them. Verifier value-add layered ON a working
   exploration loop, instead of a goal-induction bottleneck. (Five levers nulled trying to *fix*
   goal-grounding directly; the answer is to *demote* it, not fix it.)

**Our differentiation vs a pure copy** (so this is energy-augmented, not cloning the leader):
the leader resets to *random* each level, discarding cross-game transfer. We have a useful cross-game prior —
**warm-start the online CNN from the prior and reset to the PRIOR (not random) on level-up.** Prior +
online-adapt should dominate random-init online. This is the `arc-energy-augmented-strategy` move
(objective/transferable structure augmenting the winners' per-game learners).

---

## 5. Deadline-aware sequencing (6 days)

- **Day <1 (floor):** Lever 2 — flip the default to `CARNOT_ARC_TRUST_METRIC=cell_recall` /
  `trust_metric="cell_recall"` so our existing online CNN un-gates. Low-risk; measures whether any online
  adaptation helps even through the goal path. Pure parity-safe knob.
- **Days 1–3 (the bet):** Levers 1+3 fused as a **goal-free fallback policy** in the existing cascade
  (`cascade=True`, so it's additive, not a rip-and-replace): online binary-frame-change CNN with a coord head
  proposing clicks, per-level reset to the cross-game prior, reward-driven. Hooks at
  `_load_submitted_frame_change_scorer` (`:216`) + `_candidates` (`:720`) + an `observe/train` per step in
  `StepwiseExplorer`.
- **Measurement (falsifiable):** the `experiment_4605` held-out **color-permuted variant** harness, arms
  `{frozen, online-scratch, online-warm}`, B≈100, metric **held-out first-win rate** AND a multi-level
  deepening probe on lp85/sc25. **KILL** if online-warm does not beat frozen by ≥+0.05 first-win.
- **After the deadline (the moat):** the verifier-as-pruner layer on top of the working exploration loop.

---

## 6. Risks (honest)

- **Latency / no GPU.** The Kaggle validation env is ~16GB VRAM and the run has a 12h / 600-RPM cap. The
  leader runs 8h wall-clock with thousands of online steps. A `SmallFrameChangeCNN` Adam step every 5 actions
  is tiny and likely fine on CPU, but **must be wall-clock-measured** before committing — the synth flagged
  "no GPU; confirm CPU latency."
- **Per-level reset discards transfer.** Mitigated by resetting to the cross-game *prior*, not random — but
  that hybrid is unproven; the A/B must include both reset targets.
- **Architecture change risk.** Add the leader loop as a **parallel/fallback** policy in the cascade for the
  deadline; do NOT rip out the goal-induction path (it may still win where a goal *is* cleanly inducible).
- **Frozen generator unaffected.** This is the action-effect CNN, not the LLM. Qwen3.5-9B-MTP stays frozen
  (`project_arc_live_generator`).
- **Live-path reachability.** Any new module must be in the scored agent's import closure
  (`arc_orphan_solver_lint`); hooking into `E3AgentPolicy`/`StepwiseExplorer` satisfies this by construction.

---

## 7. Bottom line

The 15× gap is **online action-learning**, confirmed in code. The workflow's #1 lever (online re-ranker)
buys first-win but **not** the multi-level wall — because we keep goal-induction as the driver. The move that
buys **both** is to adopt the leader's **goal-free, reward-driven, online-exploration loop with a CNN
coordinate head**, demoting goal-induction + the energy verifier to a **router/pruner** (the north star),
and differentiating by warm-starting the online CNN from our cross-game prior. The multi-level wall is not
crossed by *fixing* goal-grounding (5 nulled levers) — it is crossed by *demoting* it.
