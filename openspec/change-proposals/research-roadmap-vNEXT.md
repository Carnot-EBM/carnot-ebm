# Research Roadmap — Milestone 2026.06.423

**CLOSE THE LIVE-SUBMISSION GAP (33 → 53): bank replayable trajectories + refresh the validated package + env-adaptive re-solve for version-drift — the highest-confidence score lever — flanked by the feature-ROUTER for seen→hidden transfer (the operator's 2026-06-21 frontier)**

- **Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-22.
- **Milestone doc for:** `research-roadmap-next.yaml` (2026.06.423).
- **Sprint:** ARC-AGI-3 submission sprint through **2026-06-30** (CLAUDE.md
  "ARC-AGI-3 Submission Sprint Forcing Function" — majority ARC; ≥1 level-up
  bank; 2 reserved infra; 1 per-board hardware; 1 SOTA-ingestion; all
  experiments `codex`/`gpt-5.5`; planner+retro stay Claude Opus).

---

## 1. What the previous milestone (.422) proved

`.422` ("ACTION EFFICIENCY — ship the leaderboard's #1 lever") was, like `.418`–`.421`,
**a near-total honest null on the held-out headlines, with one real capability bank.**

| Phase | Result | Verdict |
|---|---|---|
| A1 clickability/action-effect CNN predictor as a RANKER (exp4568) | `actions_delta=0.0` — NO efficiency gain (2nd null after `.416) | `complete: clickability_predictor_no_efficiency_gain_honest_null` |
| A2 verifier-guided frontier EXPANSION (exp4569) | `transfer_delta=-0.04` (WORSE) **with a BROKEN positive control** (`positive_control_passed=None`, false-negative-risk OPEN) | not a clean null — a broken test |
| A3 level-up bank (exp4570) | **cn04 L2 banked → `reproducible_total_levels` 52→53** | ✅ the one real win |
| A4 ka59 hidden-field state probe (exp4571) | no bank; disambiguation control passed | `complete: ..._gap_sharpened_no_bank` |
| A5 integration (exp4572) | `no_lever_raises_a_metric` (heldout_solve_rate 0.04, unchanged) | honest null |
| A6 PersistentAEM persist + transfer (exp4573) | `actions_reduced=1.0` on dc22/m0r0/ka59 **but only over a cached pool, no new bank** | `success: ..._value_added` (ordering-only) |
| B1/B2/C/D | action-efficiency co-headline metric wired; learned-CNN DURATION guard; hardware audit; action-effect SOTA ingested | shipped |

**The triply-confirmed diagnosis (`.421 A6 → `.422 A1 → `.422 A6):** the bottleneck
is **candidate GENERATION, not ranking.** A re-ranker / predictor-ordering of a fixed
pool adds 0 value on first-contact because **the winning candidate is never generated.**
Ranking works only when the winner is already in the pool (A6 cached-pool ordering).

**Two decisive signals the planner must not re-derive (operator + outer-loop, 2026-06-21):**

1. **`reproducible_total_levels` is partly a "mirage" for the leaderboard** (GAP-LIVE-INTEGRATION,
   operator 2026-06-19 #1, the *highest* score-mover): we have **53 reproducible levels but the
   2026-06-21 live scorecard submitted only 33.** The submitted agent ships bare BFS; the stronger
   stack is not wired; the gap is mechanical — a STALE package (sc25 L5, tu93 L5, lp85 L5, and
   cn04/cd82/sp80/su15/m0r0 L2 banked since the submission are NOT in it), **9 graph-explore /
   config-rule solves with NO banked replayable trajectory** (dc22/ft09/g50t/s5i5/sb26/vc33/re86/
   bp35/lf52), and **sc25 live version-drift** (`env_match=false`, hash 635fd71a). Closing it is
   "integration, not modeling."
2. **value-head best-first EXPANSION (the `.422 A2 approach) REGRESSED this weekend**
   (goal-bias best_first 0.0152; `value_weight>0` 2/11 vs the SHIPPED diversity floor's 4/11).
   Do **not** re-run verifier-guided expansion as a headline. The operator's freshest direction
   (2026-06-21) is the **feature-ROUTER** (mechanic-from-early-play → approach) + **diversity-floor
   transfer** + **self-play every milestone**.

## 2. The `.423 strategy — two thrusts, both genuine sprint progress

**Thrust 1 (HEADLINE, highest-confidence score lift): close the 33→53 live-submission gap.**
This is the literal sprint output (a bigger submittable package) and the operator's standing
"beat 33 levels" gate. The work is concrete and achievable: (a) refresh the validated package to
current banked depth, (b) bank a replayable action trajectory for every offline-reproduced level
that lacks one (extract the winning sequence the `reproduce()` gate already replays), (c) make the
submit replay **env-adaptive** (discover-from-env re-solve, not flat-replay) for version-drift games
like sc25. Submission stays operator-only; the task PREPARES + offline-validates and emits
`ready_for_operator_submit`.

**Thrust 2 (FRONTIER, the operator's 2026-06-21 direction): the feature-ROUTER for seen→hidden
transfer.** Classify a game's mechanic from EARLY-PLAY features → route to the matching general
approach already built into the toolkit (diversity-on-stall, systematic BFS, goal-distance A*,
LLM-reasoner). This is the general seen→hidden transfer per-game recipes cannot do, learned from the
self-play loop's pos/neg traces — the literature instantiation is SkillRouter (arXiv:2603.22455) /
SkillGraph (arXiv:2605.12039). Measured on the variant proxy, reporting ACTION cost (per the
weekend best-first-regression heads-up), with a working positive control.

These are flanked by the mandated slots: the level-up GUARANTEE via the self-play loop (grows
`reproducible_total_levels`), diversity-floor transfer validation, self-learning persist+reuse,
integration into the submitted agent, 2 reserved infra, 1 hardware-continuity, 1 SOTA-ingestion,
capstone.

## 3. The three biggest gaps between current state and the PRD/north-star vision

1. **The submitted score lags the banked capability (33 vs 53).** The north star is ARC-AGI-3
   solve-rate; we have already-earned public-game levels not in the submission because of stale
   packaging, missing replay trajectories, and live env-drift. (→ A1, A6, B1.)
2. **Zero proven seen→hidden generalization.** `generic_transfer` has sat at 0.04 for five
   milestones; every learned-signal lever (verifier-router, re-induction, clickability-ranker,
   value-head expansion) nulled because it cannot GENERATE the unseen winner. The untried lever is
   the feature-ROUTER (route to a general *generator/exploration* approach by mechanic class) + the
   SHIPPED diversity floor's exploration broadening. (→ A3, A4.)
3. **The live discovery loop is still partly frozen-replay.** The "Live Hidden-Game Discovery Agent"
   framing (CLAUDE.md, MANDATORY) says the deliverable is the runtime discovery PROCESS, not a frozen
   trajectory. sc25's live miss (`env_match=false`) is exactly the frozen-replay failure mode; an
   env-adaptive re-solve is both the score fix AND a step toward the genuine deliverable. (→ A1, A5.)

## 4. Architecture (what changes this milestone)

```
                          ┌──────────────────────────────────────────────┐
   LIVE ARC-AGI-3 env ───▶│  SUBMITTED AGENT  (E3AgentPolicy)             │
   (25 public + hidden)   │  arc_competition_agent.py                     │
                          │   • StepwiseExplorer + diversity floor (shipped)│
   A1: env-adaptive ─────▶│   • A1: env-adaptive re-solve (not flat-replay)│──▶ live-submittable
       re-solve +         │   • A3: feature-ROUTER → approach selection    │     level count
       trajectory bank    │   • A6: winners wired into SUBMITTED_AGENT_CONFIG│   (B1 honest metric,
                          └───────────────┬──────────────────────────────┘     33 → target >33)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                   ▼
  arc_solve_registry.yaml          arc_solver_kit.py                  arc_solve_learning.py
   (per-game mechanics +            (reusable primitives +             (recommend_approach +
    reproduce() gate;               reproduce gate; A5 persists        A3 EXTENDS with the
    A2 grows 53→54+)                the winning primitive)             early-play mechanic router)
        ▲                                 ▲                                   ▲
        │      A2 self-play loop (arc_loop_solve --auto): solve → reproduce → │
        └──────  train+checkpoint learned verifier on pos/neg traces  ───────┘
                          (the self-improvement engine, every milestone)
```

The verifier stays oracle-DISTINCT where it claims value (`verifier_is_oracle: false`); A1/A3/A4
are exploration/generation/routing levers measured by action cost + first-win + live-submittable
count, NOT circular execution wins.

## 5. Phases & tasks (12 tasks: exp4579–exp4590)

| id | phase | track | what | gate |
|---|---|---|---|---|
| exp4579 | 0 | transition | archive .422 → activate .423; record .422 close-state | YAML parses + pre-test green |
| exp4580 | A1 | arc-north-star (HEADLINE) | live-submission gap: refresh package + bank missing trajectories + env-adaptive re-solve | offline-repro-gated live-submittable count **>33** |
| exp4581 | A2 | arc-north-star (LEVEL-UP GUARANTEE) | self-play loop `arc_loop_solve --auto`: bank +1 on a not-recently-deepened game + train verifier | `reproducible_total_levels` 53→54+ |
| exp4582 | A3 | arc-north-star (FRONTIER) | feature-ROUTER: mechanic-from-early-play → approach (SkillRouter-style) | `generic_transfer`>0.04 w/ CI + action cost + control |
| exp4583 | A4 | arc-north-star | diversity-floor held-out transfer validation (+ bank any new win) | first-win count up vs diversity-off + control |
| exp4584 | A5 | arc-north-star (SELF-LEARNING) | persist the winning primitive (env-adaptive re-solve / router) + cross-game transfer | primitive persisted + transfer characterized |
| exp4585 | A6 | arc-north-star (INTEGRATION) | wire winners into SUBMITTED_AGENT_CONFIG; re-measure; parity green | a real metric rises, parity green |
| exp4586 | B1 | infra (reserved) | LIVE-SUBMITTABLE level count as a capstone co-headline metric (vs the mirage) | metric wired + asserting tests |
| exp4587 | B2 | infra (reserved) | METHODOLOGY_MISSING / offline-arc-substrate reader guard | not-flagged-on-offline + still-flagged-on-fake-LLM |
| exp4588 | C | hardware | per-board continuity (KV260 SSH / GateMate USB / PolarFire SSH) | per-board reachability recorded |
| exp4589 | D | sota-ingestion (reserved) | skill-routing / skill-library + env-adaptive replay SOTA → A1/A3 mapping | ≥3 methods w/ real arXiv IDs + flag for .424 |
| exp4590 | E | capstone | scorecard: did A1 raise live-submittable >33? A3 raise transfer? A2 grow levels? | all co-headline metrics reported |

## 6. Dependency graph

```
exp4579 (phase0)
   └─▶ exp4580 (A1 submission gap) ─┐
   └─▶ exp4581 (A2 self-play bank)  ├─▶ exp4584 (A5 persist winner)
   └─▶ exp4582 (A3 feature-router)  ├─▶ exp4585 (A6 integration) ─▶ exp4590 (E capstone)
   └─▶ exp4583 (A4 diversity xfer) ─┘                                    ▲
   └─▶ exp4586 (B1 submittable metric) ───────────────────────────────┤
   └─▶ exp4587 (B2 substrate guard) ───────────────────────────────────┤
   └─▶ exp4588 (C hardware) ────────────────────────────────────────────┤
   └─▶ exp4589 (D SOTA ingestion) ──────────────────────────────────────┘
```

A5/A6/E read A1–A4 artifacts; B1 feeds E's co-headline reporting. All ARC tasks are
INDEPENDENT of each other's success (the level-up guarantee A2 holds even if the headlines null).

## 7. Hardware requirements

- **A1–A6, B1, B2, D, E:** CPU / iGPU only — offline ARC arcade (`environment_files`, zero
  quota), cached corpora, learned-verifier checkpoints. No 3090s, no live LLM at submit-time
  (Mode-1 replay / env-adaptive re-solve). If A6's integrated run invokes the LLM proposer it
  declares `live_llm_inference` + the Qwen3.5-9B-MTP precondition (the frozen sprint generator,
  iGPU, NEVER the 3090s — [[project_arc_live_generator]]).
- **C (hardware continuity):** KV260 via SSH ONLY (never host SD-card), GateMate USB detect,
  PolarFire SSH — per-board reachability audit, no new bitstream build.
- **Live submission:** OPERATOR-ONLY (external publication). The sprint tasks PREPARE + offline-
  validate and emit `ready_for_operator_submit`; they never close a scorecard.

## 8. Disciplines honored

- **ARC-AGI-3 Submission Sprint Forcing Function** (majority ARC, monotonic `reproducible_total_levels`,
  codex experiments, Opus planner/retro) · **ARC Level-Up Attempt Guarantee** (A2 dedicated bank;
  A4 possible second) · **ARC-AGI-3 Incremental-Progress Scoping** (+1..+n on ONE game, never "full
  solve") · **ARC Solve Reproducibility + Solver-Reuse Discipline** (every solve `offline_reproduced`-
  gated; A5 persists into `arc_solver_kit`/registry) · **Live Hidden-Game Discovery Agent framing**
  (A1 moves from frozen-replay toward env-adaptive discovery) · **Circularity/Oracle-Distinctness**
  (`verifier_is_oracle: false` on all value claims) · **Failed-Experiment Rerun Discipline**
  (A3/A4 carry `prior_failures` with all four sub-fields; legit continuations carry `operator_override`)
  · **Pre-Launch Preconditions** (every compute-bound task has a PRECONDITIONS step) · **Principle-
  Annotated Artifact Fields** · **Verdict Terminal-Prefix** · **Adversarial Artifact Verification +
  FALSE_NEGATIVE_RISK guards** (working positive controls after the `.422 A2 broken-control trap) ·
  **Reserved infra (2) + hardware (1) + SOTA-ingestion (1)** · **SOTA-Ingestion Cycle** (reliable
  channel, no `/deep-research` in-loop).
