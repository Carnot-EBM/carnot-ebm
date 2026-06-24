# Research Roadmap — Milestone 2026.06.432

**Status:** PROPOSED (outer-loop Claude Opus 4.8 planner, 2026-06-24)
**Theme:** THE WALL IS THE EXPLORER'S ACTION-PROPOSAL DISTRIBUTION AT L1 FIRST-CONTACT — not selection,
not generation-by-search. The 7th-consecutive bridge-not-crossed milestone (`.431) ran the decisive
diagnostic and **CONFIRMED `wall_diagnosis=l1_first_contact`**: the generic agent reaches first-win on only
1/25 public games (`first_win_rate=0.04`). Both `.431 generation-by-search levers nulled. `.432 PIVOTS to
**DIRECTED EXPLORATION** — reshape the explorer's action-proposal distribution so a winning L1 trajectory
APPEARS in the pool: **controllable-novelty E3 proposal policy** (A1, HEADLINE) and **program-synthesis
action-effect proposal filter** (A2).
**Sprint:** ARC-AGI-3 Submission Sprint (CLAUDE.md, through **2026-06-30** — 6 days to the ARC Prize
Milestone #1 deadline, $25K).
**Predecessor:** 2026.06.431 (`research-roadmap.yaml`, capstone exp4686).

---

## 1. What the previous milestone (.431) proved

`.431 pivoted from SELECTION to CANDIDATE GENERATION and attacked the wall with two structural
generation-by-search methods, gated on a STEP-1 decisive diagnostic. The capstone (exp4686) verdict:
**`capability_grew_59_to_60`** — A3 banked a new reproducible level, but **`bridge_crossed_for_solve = FALSE`
for the 7th consecutive milestone**, and BOTH headline mechanisms nulled on live SOLVE-RATE.

| Task | Lever | Verdict | Outcome |
|---|---|---|---|
| A1 (exp4676) | Hierarchical subgoal search over the live E3 frontier (STEP-1 diagnostic first) | `complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating` | **NULL — but the diagnostic DECIDED THE OPEN QUESTION.** `wall_diagnosis=l1_first_contact`; `generic_first_win_by_config` ⇒ first-win 0.04 (only lp85 of 25). The subgoal build then nulled: `generic_agent_reached_level=0`, `subgoal_decomposition=[]`, residual `value_head_still_not_separating`. `chosen_submitted_config=unchanged`. |
| A2 (exp4677) | PoE-World factored-executable subgoal planner | `complete: poe_world_factored_planner_no_coverage_gain_residual_logged` | **NULL.** `candidate_generation_coverage_factored=0.0`, `coverage_delta=0.0`, `first_win_rate_delta=-0.04`, residual **`experts_overfit_prefix`** (every induced expert failed held-out transition trust → none composed). `chosen_submitted_config=unchanged`. |
| A3 (exp4678) | Level-up self-play (rotation target sb26) | `success: sb26_L2_offline_reproduced` | **BANKED** sb26 L2 → `reproducible_total_levels` 59→60. Learned verifier checkpointed. |
| A4 (exp4679) | Submission-package refresh | `success` | `live_submittable_level_count=60` (>33), `ready_for_operator_submit=True`. (OLD framing — retargeted in `.432, see §below.) |
| Capstone (exp4686) | Scorecard + G1–G4 | `complete: capability_grew_59_to_60`, `paper_ready=True` | FoVer 0.9131 frozen; `bridge_crossed_for_solve=False`. |

### The decisive new input: the wall is the PROPOSAL distribution, upstream of BOTH selection and search

`.431's A1 STEP-1 diagnostic resolved the long-running 0.04-vs-0.59 first-win ambiguity in favor of **0.04**
and pinned the binding wall as **L1 first-contact**: the generic `E3AgentPolicy` fails to reach L1 on 24/25
public games. Then its two generation-by-search levers both nulled — because they operate *over* a candidate
pool that never contains a winning L1 trajectory:

1. **Hierarchical subgoal search nulled because there is nothing to decompose.** `subgoal_decomposition=[]` and
   `generic_agent_reached_level=0`: you cannot chain reachable subgoal legs into a winner when no leg toward
   the winner is ever proposed. Residual `value_head_still_not_separating` is the symptom, not the cause.

2. **PoE-World factored planning nulled because the experts overfit the prefix.** `experts_overfit_prefix`:
   every induced object-expert failed held-out transition trust, so the product model had no replay-stable
   factor to plan with. Composing planners over a degenerate proposal stream produces a degenerate plan.

3. **The convergent residual across ≥5 persisted primitives, verbatim.** `ops/arc_solve_registry.yaml`
   `transfer_dead_ends`: *"If the winning action is absent from the candidate group, [the operator] can only
   reorder generated candidates; candidate generation remains the residual bottleneck."*

**Synthesis (the `.432 thesis).** Six selection levers (`.425–`.430) and two generation-by-search levers
(`.431) all nulled because they all act *after* the explorer proposes actions. The operator named the wall
2026-06-22 (*"make-a-winner-appear, not select"*); the `.431 D ingestion
(`docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md`) sharpened it to its root: the
winner is not in the pool because **the explorer's PROPOSAL distribution does not cover it.** Its explicit
*"bottom line for the `.432 roadmap"* is to build **directed proposal coverage** — change *what the explorer
proposes* so a winning L1 trajectory APPEARS, before any selection or search can help.

---

## 2. The three biggest gaps (current state vs. north star)

The north star (`ops/north-star.md` §0) is the LIVE agent self-discovering hidden-game solves accurately and
efficiently. The gaps, ranked:

1. **PROPOSAL — the explorer never proposes the winning L1 trajectory (HEADLINE GAP).** At 0.04 generic
   first-win, the binding constraint is upstream of selection AND of generation-by-search: the action-proposal
   distribution is too narrow / blind to surface a winner on 24/25 games. The fix is to reshape the proposal
   distribution: a CONTROLLABLE-novelty intrinsic bonus (NGU + RND) that drives exploration toward actions
   whose *effects* are new (not cosmetic), plus a held-out-validated action-effect program filter that prunes
   blind spatial sweeps to mechanically-relevant actions.

2. **The dev-proxy-vs-scored conflation in the readiness gate.** PHASE-A4 gated readiness on
   `live_submittable_level_count > 33` — the depth of the offline REPLAY package — but the replay path scores
   **~0** on the hidden leaderboard (first scored submission = **0.08**, ref 53862349). The operator's
   2026-06-24 directive retargets A4 to the only offline proxy that tracks the scored lane: the
   `experiment_4605` HELD-OUT generic first-win on color-permuted variants (bootstrap-CI > 0).

3. **Per-game exploration may not TRANSFER (the next wall).** Even if directed exploration lifts first-win on
   a public game, the scored target is HIDDEN OOD games. If the explorer re-derives novelty from scratch on
   each game, it does not generalize. The `.432 D ingestion scopes the AMORTIZED / transferable exploration
   fallback (learned exploration prior, Go-Explore archive) for `.433.

---

## 3. Architecture — where `.432 acts

```
                          THE LIVE ARC AGENT (E3AgentPolicy — the SCORED deliverable)
                          ════════════════════════════════════════════════════════
  frame ──▶ PERCEPTION ──▶ ACTION PROPOSAL ──▶ CANDIDATE POOL ──▶ SEARCH/SELECT ──▶ plan_in_model ──▶ REPLAY GATE
            (v3 feats,     (StepwiseExplorer   (the pool the      (.425–.431       (induced          (arc_solver_kit
             LOO 0.725      — blind/value-      winner is NOT      levers — all     dynamics+goal)    .reproduce)
             — DONE)        ranked sweep)        in)               nulled)
                                 ▲
                                 │  ◀── .432 A1 (HEADLINE) ── CONTROLLABLE-NOVELTY PROPOSAL POLICY
                                 │     episodic kNN + RND lifelong novelty over a CONTROLLABLE-novelty
                                 │     embedding (frame-delta + action-EFFECT; controllability gate =
                                 │     the .427 noisy-TV fix); intrinsic proposal bonus BEFORE ranking
                                 │
                                 │  ◀── .432 A2 ── PROGRAM-SYNTHESIS ACTION-EFFECT PROPOSAL FILTER
                                 │     induce per-game action→effect programs; REJECT on held-out
                                 │     transitions (the experts_overfit_prefix fix); prune proposals
                                 │     to mechanically-relevant actions
                                 ▼
                 THE .425–.431 WALL WAS DOWNSTREAM ── select/search a pool that never contained the winner
                 THE .432 PIVOT                    ── widen/sharpen the PROPOSAL so the winner ENTERS the pool
```

`.432 changes ONLY the live modules in the `E3AgentPolicy` import closure (`arc_competition_agent`
StepwiseExplorer, `arc_value_learner` action-effect features, `arc_frame_change_predictor`,
`arc_llm_reinduction`, `arc_executable_world_model`, `arc_solver_kit`) so `scripts/arc_orphan_solver_lint.py`
stays green and `tests/python/test_arc_submitted_agent_parity.py` stays green — the measured agent IS the
deployed agent (ARC Live-Path Reachability Discipline).

---

## 4. Phases

### Phase 0 — Transition (exp4687)
Archive `.431 → activate `.432; assert the YAML parses + the smart-subset pre-test gate is green; record the
TRUE `.431 close-state (A3 59→60; A1 wall=`l1_first_contact` + nulled; A2 coverage 0 + `experts_overfit_prefix`;
both `unchanged`; bridge_crossed=False; first scored sub=0.08; paper_ready=True). Codex, `max_turns: 30`.

### Phase A — ARC North Star (the majority; operator-mandatory)

- **A1 (exp4688) — HEADLINE: CONTROLLABLE-NOVELTY E3 PROPOSAL POLICY.** NGU (arXiv:2002.06038) + RND
  (arXiv:1810.12894) + Strategy-Guided Exploration (arXiv:2603.02045). Episodic kNN novelty + RND lifelong
  novelty over a **controllable-novelty embedding** (frame-delta + action-EFFECT features — the controllability
  gate is the noisy-TV fix for the `.427 dense-curiosity null exp4628), applied as an intrinsic proposal bonus
  on the live StepwiseExplorer BEFORE value-ranking, with a family of exploration temperatures.
  **Gate:** the GENERIC live agent reaches a NEW level on ≥1 clean L1-only game where flat exploration fails,
  offline-reproduced, with a **NO-NOVELTY-BONUS** ablation AND a **COSMETIC-NOVELTY** (controllability-gate-off)
  ablation that do NOT. `live_llm_inference`; `verifier_is_oracle:false`; `solve_provenance:
  live_agent_self_discovery`. `prior_failures`: exp4628-a1 (dense-curiosity null), exp4676-a1 (subgoal-search
  null). Codex, `max_turns: 160`.

- **A2 (exp4689) — second independent mechanism: PROGRAM-SYNTHESIS ACTION-EFFECT PROPOSAL FILTER.** PoE-World
  (arXiv:2505.10819) + Program-Synthesis-Guided RL (arXiv:2102.11137). Induce small per-game action→effect
  programs from observed prefixes, **REJECT programs that fail held-out transitions** (the direct
  `experts_overfit_prefix` fix), and prune the explorer's primitive proposals to mechanically-relevant
  clicks/keys. **Gate:** candidate-generation coverage up (winner appears where the matched blind-proposal
  baseline did not) AND a held-out first-win lift (CI excludes baseline), offline-reproduced.
  `live_llm_inference`; `verifier_is_oracle:false`; `solve_provenance: live_agent_self_discovery`.
  `prior_failures`: exp4677-a2 (PoE-World `experts_overfit_prefix`), exp4653-a2 (energy-fitness QD `no_winner`).
  Codex, `max_turns: 150`.

- **A3 (exp4690) — LEVEL-UP GUARANTEE + self-play.** Bank +1 reproducible level (60→61) on a rotated clean game
  (PREFER bp35/re86/s5i5/g50t/r11l/lf52 L1→L2; SKIP sb26/dc22/vc33/ft09/ls20/sk48/ka59/wa30/cd82-L3/sp80-L3/su15-L3)
  AND train+checkpoint the learned verifier on pos/neg traces. INDEPENDENT of A1/A2 so the guarantee holds.
  `verifier_ensemble_against_cached_candidates`; `solve_provenance: development_proxy`. `prior_failures`:
  exp4618-a3 (sk48 no-bank). Codex, `max_turns: 150`.

- **A4 (exp4691) — SCORE, RETARGETED (2026-06-24 operator directive).** Gate readiness on the HELD-OUT generic
  first-win lane (`experiment_4605` `first_win_rate_integrated` vs the 0.04/0.08 last-submission baseline,
  bootstrap-CI lower bound > 0), NOT replay-package depth. Keep the reproduced replay package as a FLOOR
  artifact only and strip the "honest leaderboard score" framing. `operator_override` cites the directive.
  Codex, `max_turns: 120`.

- **A5 (exp4692) — persist + transfer.** Persist `.432's winning directed-exploration primitive
  (controllable-novelty proposal operator OR the program-synthesis action-effect filter) into `arc_solver_kit`
  + the registry; measure cross-game transfer (characterize the null honestly if value-null). Codex,
  `max_turns: 100`.

- **A6 (exp4693) — integration.** Fold the winning A1/A2 config into `SUBMITTED_AGENT_CONFIG` (single source of
  truth); re-measure the integrated HELD-OUT first-win + deepen-rate on the scored agent (the A4-retargeted
  lane); keep parity green; avoid the `.430 A6 TAUTOLOGY when `unchanged`. Codex, `max_turns: 100`.

### Phase B — Infra (2 reserved slots)

- **B1 (exp4694) — L1-first-contact PROPOSAL-COVERAGE CI-metric + honest first-win floor.** A metric/gate that
  measures whether the explorer's action-proposal distribution REACHES the winning L1 trajectory (the
  proposal-stage analog of the `.431 generation-coverage gate), re-affirm the honest 0.04 first-win floor
  (a permissive harness cannot silently inflate it), and a proposal-coverage floor. Unit tests. Codex,
  `max_turns: 100`.

- **B2 (exp4695) — adversarial_verify hardening.** Two guards for the `.432 lever class:
  (1) NOVELTY-PROPOSAL-WITHOUT-ABLATION (a controllable-novelty win must report the no-novelty AND
  cosmetic-novelty ablations strictly lower + offline_reproduced); (2) PROPOSAL-FILTER-WITHOUT-HELDOUT-REJECTION
  (a coverage-up claim must report the held-out rejected-program count + the matched blind-proposal baseline).
  Honest artifacts not flagged; unit tests. Codex, `max_turns: 80`.

### Phase C — Hardware continuity (1 per-board slot) (exp4696)
Per-board reachability audit (KV260 via `ssh kria` — SSH-only, NEVER host SD-card; PolarFire via `ssh
polarfire`; GateMate via `openFPGALoader -c dirtyJtag --detect`). No bitstream build, no fabric-acceleration
claim. Codex, `hardware_smoke`, `max_turns: 60`.

### Phase D — SOTA-ingestion → `.433 (exp4697)
Focused literature pass on the NEXT fallback: AMORTIZED / TRANSFERABLE exploration (learned exploration prior,
meta/in-context exploration, Go-Explore return-then-explore archive) — the deeper wall if per-game directed
exploration works but does not transfer to hidden OOD games. Reliable sweep + WebSearch/WebFetch only;
`/deep-research` BANNED. Codex, `max_turns: 60`.

### Phase E — Capstone `.432 (exp4698)
Aggregate the scorecard + the HEADLINE DECISION: did DIRECTED EXPLORATION cross the offline→live bridge for
L1-first-contact (A1 generic new level with both ablations failing; A2 coverage up + held-out first-win lift
with held-out rejection run and CI excluding baseline; A3 bank 60→61; A4 retargeted held-out first-win vs
0.04)? Skip flagged / control-failed / ablation-missing artifacts. Confirm `verifier_is_oracle:false` on every
value claim; re-affirm G1–G4 `paper_ready` (FoVer 0.9131 frozen); submission operator-only. Codex,
`max_turns: 40`.

---

## 5. Dependency graph

```
exp4687 (Phase 0, transition)
   │
   ├─▶ exp4688 (A1 controllable-novelty proposal) ──┐  HEADLINE — confirmed l1_first_contact target
   ├─▶ exp4689 (A2 program-synthesis filter) ───────┤  reads A1's target game if present
   ├─▶ exp4690 (A3 level-up self-play 60→61) ────────┤  INDEPENDENT (guarantee holds even if A1/A2 null)
   │                                                  ▼
   ├─▶ exp4691 (A4 held-out first-win readiness, RETARGETED)
   ├─▶ exp4692 (A5 persist+transfer of the winning primitive) ◀── reads A1/A2
   ├─▶ exp4693 (A6 integration into SUBMITTED_AGENT_CONFIG) ◀──── reads A1/A2 chosen_submitted_config
   ├─▶ exp4694 (B1 proposal-coverage CI-metric) ◀──────────────── reads A1 coverage fields
   ├─▶ exp4695 (B2 adversarial_verify hardening) ◀─────────────── reads A1/A2 artifacts as fixtures
   ├─▶ exp4696 (C hardware continuity)         [independent]
   ├─▶ exp4697 (D SOTA-ingestion → .433)       [independent]
   │
   └─▶ exp4698 (E capstone .432) ◀───────────── aggregates ALL upstream (summarize_artifact.py)
```

No `requires:` chain references a retired exp_id. A2/A5/A6/B1/B2/E read A1/A2 artifacts opportunistically
(IF PRESENT) — none hard-blocks, so a single null does not cascade.

---

## 6. Hardware requirements

- **A1/A2 (live_llm_inference):** the frozen live generator **Qwen3.5-9B-MTP on the iGPU (NEVER the 3090s)**
  for the world-model induction / strategy-conditioned arm. Construct the proposer on a **FREE port (e.g.
  8920)** + `/props`-verify it serves Qwen (the port-8919 gemma-squat confound). PRECONDITION-gated on the
  cached GGUF.
- **A3/A4/A5/A6/B1 (offline):** offline arcade + cached candidates, zero quota, no live game server.
- **C (hardware_smoke):** KV260 (`ssh kria`), PolarFire (`ssh polarfire`), GateMate (`openFPGALoader`) —
  reachability only.
- **B2/D/E (aggregation):** read upstream artifacts / linter edits / literature; no model load.

---

## 7. SOTA references incorporated (all arXiv HTTP-200 verified in exp4685)

| arXiv | Title | Used by |
|---|---|---|
| 2002.06038 | Never Give Up: Learning Directed Exploration Strategies | A1 (episodic + lifelong novelty) |
| 1810.12894 | Exploration by Random Network Distillation | A1 (RND lifelong novelty) |
| 2603.02045 | Expanding LLM Agent Boundaries with Strategy-Guided Exploration | A1 (strategy-conditioned arm) |
| 2505.10819 | PoE-World: Compositional World Modeling with Products of Programmatic Experts | A2 (action→effect programs) |
| 2102.11137 | Program Synthesis Guided RL for Partially Observed Environments | A2 (proposal pruning) |
| 2005.05960 | Planning to Explore via Self-Supervised World Models | D → `.433 (support arm) |
| 2502.10077 | Empowerment Gain through Causal Structure Learning in Model-Based RL | D → `.433 (support arm) |
| 1712.06560 | Novelty-Seeking Population for Exploration in ES | D → `.433 (QD archive arm) |

Spec note: `docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md` (the `.431 D ingestion,
whose "bottom line for the `.432 roadmap" this milestone executes).

---

## 8. Discipline compliance

- **ARC sprint forcing function (through 2026-06-30):** majority ARC (A1–A6); ≥1 level-up attempt that BANKS a
  level (A3, lint OK 3≥1); self-play every milestone (A3 trains+checkpoints the verifier); 2 reserved infra
  (B1/B2); 1 per-board hardware (C); 1 SOTA-ingestion (D). All experiments `codex`/`gpt-5.5`; planner/retro
  stay Claude Opus.
- **ARC Live-Path Reachability:** all A1/A2 changes are in the `E3AgentPolicy` import closure;
  `arc_orphan_solver_lint` + `test_arc_submitted_agent_parity` stay green.
- **Circularity:** `verifier_is_oracle:false` on every value claim (the novelty bonus / induced programs /
  value head are oracle-distinct from the executable reproduction win-check).
- **solve_provenance:** A1/A2 `live_agent_self_discovery`; A3 `development_proxy` (honest dev twin).
- **Failed-Experiment Rerun:** A1/A2/A3 carry complete `prior_failures` blocks (all 4 sub-fields +
  `retire_if_same_verdict: true`); none cite a retired-and-requires id. Routine continuations
  (phase0/A4/A5/A6/B1/B2/C/D/E) carry `operator_override`.
- **Pre-Launch Preconditions:** every compute-bound task has a PRECONDITIONS step-0 (Qwen cache + free-port
  /props guard for the live arms; offline arcade for the rest).
- **Publication gate (frozen):** G1–G4 re-affirmed at capstone; FoVer 0.9131 NEVER substituted.
- **Submission operator-only:** A4 prepares + measures readiness; capstone `leaderboard_submission=false`.

---

**Bottom line.** Seven milestones proved that selecting or searching a candidate pool cannot help when the
winner is not in the pool, and the `.431 diagnostic pinned the cause: at 0.04 generic first-win, the
explorer's action-proposal distribution does not cover a winning L1 trajectory on 24/25 games. `.432 attacks
the proposal distribution itself — widen it toward CONTROLLABLE novelty (A1) and sharpen it with
held-out-validated action-effect programs (A2) — so a winner finally APPEARS in the pool. The gate is honest:
a GENERIC new level with both ablations failing, offline-reproduced, on the retargeted held-out first-win
lane that actually tracks the scored leaderboard. 6 days to the deadline.
