# Research Roadmap v435 — STOP SELECTING, START ACTING: the leader's online action-learning driver (validly tested at last) + hypothesis-driven active-probe disambiguation — the two "make the winner appear/separable by ACTING" levers, after .434 proved STATIC selection is exhausted

**Milestone:** 2026.06.435
**Planned by:** outer-loop (Claude Opus 4.8 planner, 2026-06-25)
**Status:** PROPOSED
**Sprint:** ARC-AGI-3 submission sprint — **5 days to the ARC Prize Milestone #1 deadline (2026-06-30, $25K)**

---

## 0. TL;DR

.434 was the **10th consecutive `bridge_crossed_for_solve=False` milestone**. Its decisive new
input was **negative and clarifying**: the .434 A2 surfacing experiment (exp4713) proved that the
winning L1 candidate — now **PRESENT** in the object-centric proposal pool (coverage 1.0, the .433
perception win) at rank ~59 — is **NOT separable from same-depth distractors by ANY static
oracle-distinct feature** (`precision_at_k_delta=0.0`; the ranker even *worsened* ranks:
`[59,161,12,77] → [87,102,74,36]`). Three further levers nulled the same way: A1 (perception-grounded
L2 alignment goal) hit `alignment_under_determined`; A4 (the leader's online driver) returned
**all three arms at exactly 0.04** — a `TAUTOLOGY`/no-op signature, `flagged_adversarial` — strongly
implying it **tested dead code again** (like the .433 Go-Explore `(1,64,64)` and CNN dict-candidate
bugs); A3 banked bp35 L2 but via a `development_proxy` adapter, not live self-discovery.

**The diagnosis is now unambiguous: STATIC SELECTION / RANKING OF A FIXED POOL IS EXHAUSTED.** Ten
milestones of selection/perception/goal levers have not moved generic first-win off 0.04 (1/25). The
winner is present but unrankable. The only directions left that the project has *not* validly tested
both **ACT to change the situation** rather than rank a frozen pool:

1. **The leader's goal-free ONLINE ACTION-LEARNING DRIVER** (StochasticGoose #1, ~1.21 vs our 0.08) —
   operator-called "the highest-leverage ARC lever for the deadline" (2026-06-24). It has **never been
   validly tested**: both the .433 and .434 attempts ran degenerate/dead code (the all-arms-identical
   `TAUTOLOGY`). **.435 A1 tests it for the FIRST time**, gated on PROVEN per-arm non-degeneracy before
   any lift claim.
2. **Hypothesis-driven ACTIVE PROBING** (the .434 capstone + SOTA-ingestion hand-off; arXiv:2506.01876
   + arXiv:2309.08477) — the agent **ACTS to disambiguate** the goal/dynamics so the present-but-
   unseparable winner becomes distinguishable by its **CONSEQUENCE**, not by a static feature. **.435 A2**.

Plus: A3 banks +1 via the self-play loop (Level-Up Guarantee + verifier-train-every-milestone), and
the **five operator-mandated / carried-forward tasks that a poison pre-test SKIPPED in .434** are
re-run — most importantly **B1, the operator-MANDATORY (2026-06-25) silent-bug audit** of the .428–.434
generation-lever nulls (which formally adjudicates whether the .434 A4 TAUTOLOGY was dead code and must
reopen) and **B2, the lever-exercise-evidence guard** that mechanizes the catch.

`verifier_is_oracle:false` on every value claim; `solve_provenance` on every ARC solve; submission
operator-only; FoVer 0.9131 / G1–G4 `paper_ready` frozen.

---

## 1. What the previous milestone (.434) proved

| Phase | Lever | Verdict | What it means |
|---|---|---|---|
| A1 | Perception-grounded structural-alignment **L2 goal** (lp85) | `complete: l2_perception_goal_no_deepening_residual_alignment_under_determined` — `goal_predicate_satisfiable=False`, plan control failed | A structural piece→sprite alignment goal over detected objects is **under-determined** from one frame (which piece → which sprite is ambiguous). The 3rd L2-goal-induction null (after exp4664 single-exemplar, exp4537 reinduction). |
| **A2** | **Surface the present-but-buried winner** (oracle-distinct ranker over the coverage-1.0 pool) | `complete: surface_present_winner_no_new_level_residual_present_winner_not_separable_from_distractors` — `precision_at_k_delta=0.0`; ranks **worsened** | **THE DECISIVE FINDING.** The winner is PRESENT (coverage 1.0) but **NOT separable from same-depth distractors by any static oracle-distinct feature**. Static selection/surfacing is empirically exhausted. |
| A3 | Level-up self-play | `success: bp35_L2_offline_reproduced` | **Banked +1** (reproducible_total_levels 62→63) but `solve_provenance: development_proxy` (a GameAdapter, NOT live self-discovery). Capability grew; the bridge did not cross. |
| A4 | **Leader's online action-learning driver** (corrected) | `complete: online_action_learning_no_first_win_lift_residual_online_signal_too_sparse` — **all 3 arms exactly 0.04, `TAUTOLOGY`, `flagged_adversarial`** | **The no-op signature.** frozen=scratch=warm=0.04 to >5 sig figs ⇒ the online path almost certainly **did not actually run** (3rd dead-code suspicion in the generation track). The lever has **never been validly tested**. |
| A5 | Held-out first-win readiness | `complete: held_out_first_win_flat_no_leaderboard_change` (0.04, honest null) | First-win flat; first scored submission still 0.08. |
| A6/A7/B1/B2/C | persist / integration / **silent-bug audit** / guard / KV260 | **SKIPPED — no artifacts** | A **poison pre-test** ("1 failed, 91 passed") from an A1–A4 module cascade-skipped the entire milestone tail. The operator-MANDATORY (2026-06-25) **B1 audit** and **B2 guard** never ran. |
| D | SOTA ingestion (active-probe) | `success: …active_probe_world_model_mapped` | Mapped the .435 frontier: hypothesis-posterior active probe (2506.01876+2309.08477), epistemic-object-model MCTS (2210.13455+2601.06604), factored causal probe bank (2511.02225+2511.14262). |
| E | Capstone | `bridge_crossed_for_solve=False` (10th); `strongest_open_lever: A2_present_winner_surfacing`; `flagged_for_v435`: the active-probe methods | Hands off to **active probing**. |

**Two cross-cutting facts the .435 plan is built on:**
- **Static selection is exhausted (A2).** Do not propose another static ranker over a frozen pool.
- **The leader's loop was never validly tested (A4 TAUTOLOGY) and the audit that would confirm it (B1)
  never ran.** Both must be addressed before the lever is retired or trusted.

---

## 2. The three biggest gaps between current state and the PRD/north-star vision

1. **The generic agent cannot self-discover a FIRST-CONTACT L1 win on 24/25 games (first-win = 0.04).**
   This is the entire ARC-AGI-3 north-star (live hidden-game discovery). Every banked level above the
   1 self-discovered game is a `development_proxy` adapter, which does not transfer to hidden games.
   **Closing this is the milestone's headline.** Both .435 ARC levers (A1 online driver, A2 active probe)
   attack first-contact directly.

2. **The verifier has not yet earned its place as an ORACLE-DISTINCT contributor on ARC (north-star §5).**
   A2 (.434) showed the learned discriminative verifier cannot *statically* separate the present winner.
   The honest next form of "verifier earns its place" on ARC is **routing information-gain probes** (A2
   active-probe uses the verifier to score which action best splits the hypothesis posterior) and
   **gating online-model trust** (A1 uses the oracle-distinct frame-change CNN as a driver, not the
   exact-grid gate that gates itself out). Both keep `verifier_is_oracle:false`.

3. **Trust integrity of the project's negatives is at risk (operator-flagged 2026-06-25).** Two .433
   generation nulls and (very likely) the .434 A4 null tested dead code. The loop's core strength is its
   *trustworthy negatives*; a dead-code null masquerading as a capability limit corrupts the planner's
   reopen decisions. **B1 (audit) + B2 (mechanized guard)** close this — they are operator-MANDATORY and
   were skipped in .434, so they carry forward with priority.

---

## 3. Architecture: the L1 wall, decomposed — and where .435 acts

```
              THE L1 FIRST-CONTACT WALL  (generic first-win = 0.04 = 1/25 games)
              ───────────────────────────────────────────────────────────────
   LAYER 1  PERCEPTION / PROPOSABILITY ........................ SOLVED (.433)
            object-centric repr -> winning trajectory PRESENT in pool (coverage 1.0)

   LAYER 2  SURFACING / SELECTION (rank the frozen pool) ...... EXHAUSTED (.434 A2)
            winner present at rank ~59 but NOT separable from
            same-depth distractors by ANY static oracle-distinct feature
                                   |
                                   |  .435 stops trying to rank a frozen pool
                                   v
   LAYER 3  ACT TO CHANGE THE SITUATION  ......................  .435 (the pivot)
        +----------------------------------------------------------------------+
        |  A1  LEADER'S ONLINE ACTION-LEARNING DRIVER  (first VALID test)       |
        |      online frame-change CNN + coordinate head PROPOSES clicks;       |
        |      per-level reset to the cross-game PRIOR; self-supervised free    |
        |      labels; deepens multi-level for free.  GATE: prove arms are      |
        |      non-degenerate FIRST, THEN online-warm beats frozen by +0.05     |
        |      OR an L2 deepens offline-reproduced.   (operator highest-leverage)|
        +----------------------------------------------------------------------+
        |  A2  HYPOTHESIS-DRIVEN ACTIVE-PROBE DISAMBIGUATION  (capstone hand-off)|
        |      posterior over goal/dynamics hypotheses; take info-gain actions  |
        |      that SPLIT the posterior -> the present-but-unseparable winner   |
        |      becomes distinguishable by its CONSEQUENCE, not a static feature.|
        |      verifier scores which probe best splits the posterior.           |
        +----------------------------------------------------------------------+
                                   |
   SELF-IMPROVEMENT (every milestone)      A3  level-up self-play + verifier train (bank 63->64)
   TRUST INTEGRITY (operator-mandated)     B1  silent-bug audit (.428-.434)  +  B2  exercise-evidence guard
   READINESS / INTEGRATION                 A4  held-out first-win lane   A5  persist+transfer   A6  integration gate
   SOVEREIGNTY / SOTA / SCORECARD          C   KV260   D  SOTA ingestion (MCTS/causal probe, .436 fallback)   E  capstone
```

All ARC changes live in the **live** modules imported by `E3AgentPolicy` (`arc_competition_agent.py`,
`arc_executable_world_model.py`, `arc_llm_reinduction.py`, `arc_value_learner.py`,
`arc_frame_change_predictor.py`, `arc_live_ttt.py`) — `arc_orphan_solver_lint` and
`test_arc_submitted_agent_parity` stay green (ARC Live-Path Reachability Discipline).

---

## 4. Phase descriptions

### PHASE 0 — transition (exp4724)
Archive .434 → activate .435. Record the true .434 close-state (bridge not crossed 10th; A2
not-separable; A4 TAUTOLOGY/dead-code-suspect; A3 dev-proxy bank 62→63; **B1/B2/A6/A7/C skipped by a
poison pre-test**). **Detect + resolve the poison pre-test** so the .435 tail does not cascade-skip again.

### PHASE B1 — SILENT-BUG AUDIT (exp4725; operator MANDATORY 2026-06-25; runs EARLY) — INFRA slot 1
Carried forward (SKIPPED in .434). Audit the **.428–.434** generation/exploration-lever nulls for silent
representation no-ops (degenerate shapes, dead archives, dropped candidate pools, byte-identical arms).
**Specifically adjudicate the .434 A4 (exp4715) all-arms-0.04 TAUTOLOGY** — is it a real null or a no-op?
Emit a per-null verdict `{trustworthy_null | silent_bug_must_reopen}` + a prioritized reopen list →
`ops/arc_null_silent_bug_audit.md`. Runs early so its A4 verdict grounds A1's reopen. **No solve claims.**

### PHASE A1 — LEADER'S ONLINE ACTION-LEARNING DRIVER, first VALID test (exp4726; HEADLINE; operator highest-leverage 2026-06-24)
The goal-free online driver: an online binary-frame-change CNN with a **coordinate head that PROPOSES
clicks** (hierarchical action-then-coord sample), self-supervised free labels (`frame_changed`), Adam
lr=1e-4 BCE every ~5 actions, hash-deduped buffer, **per-level reset to the cross-game PRIOR** (the
differentiation vs the leader's random-init). **Two-stage gate (the honest fix for the dead-code history):**
(1) **prove the arms are non-degenerate** — frozen/online-scratch/online-warm must produce *distinct*
action distributions + non-zero online train-steps (NOT byte-identical first-win); if still identical →
`complete: online_driver_arms_degenerate_confirmed_harness_bug` (a *bug*, not a capability null — reopen
.436). (2) Only on non-degenerate arms: **online-warm beats frozen by ≥+0.05** held-out first-win OR a
multi-level probe (lp85/sc25) reaches L2 offline-reproduced. CPU train-step ms MEASURED (Kaggle viability).
`live_agent_self_discovery`, `verifier_is_oracle:false`.

### PHASE A2 — HYPOTHESIS-DRIVEN ACTIVE-PROBE DISAMBIGUATION (exp4727; the capstone/SOTA hand-off)
The principled answer to A2's not-separable finding. `E3AgentPolicy` keeps a small **posterior over
candidate goal/dynamics hypotheses**, asks `arc_executable_world_model` what transition each hypothesis
predicts, and chooses live actions that **maximally split that posterior** (information gain) before
committing to a solve plan — so the present-but-unseparable winner becomes distinguishable by its
**observed consequence**. The energy verifier scores which probe best splits the posterior
(`verifier_is_oracle:false`, oracle-distinct). On a hard clean L1 game where first-contact fails. **Gate:**
the generic agent reaches a NEW level via active probing, offline-reproduced, with the **no-probe (passive)
ablation FAILING** (else the win is not attributable to probing). Maps to arXiv:2506.01876 + arXiv:2309.08477.

### PHASE A3 — LEVEL-UP SELF-PLAY + VERIFIER TRAIN (exp4728; Level-Up Guarantee + self-play every milestone)
Run the standing `arc_loop_solve` loop to bank +1 NEW reproducible level on a clean game NOT deepened in
.429–.434 (PREFER first-contact L1→L2 on a hard clean public game from `re86/s5i5/g50t/r11l`; SKIP bp35
[.434], lf52 [.432], sb26 [.431], dc22 [.430], vc33 [.429], sk48 [.426], hidden-state-bound ka59/wa30,
no-grounded-delta cd82-L3/sp80-L3/su15-L3) AND **train+checkpoint the learned verifier** on pos/neg traces.
Gate: `reproducible_total_levels` 63→64+. Independent of A1/A2 so the guarantee holds even if they null.

### PHASE A4 — HELD-OUT FIRST-WIN READINESS LANE (exp4729)
Re-measure the `experiment_4605` SUBMITTED_AGENT_CONFIG held-out generic first-win over color-permuted
variants AFTER the A1/A2 levers, vs the last-submission baseline (0.04; first scored 0.08), bootstrap-CI-
excludes-0 criterion + the TAUTOLOGY null-delta markers. Parity-hard-gated.

### PHASE A5 — PERSIST STRONGEST PRIMITIVE + TRANSFER (exp4730; carried fwd, SKIPPED in .434)
Persist the strongest .435 ARC primitive (A1 online driver | A2 active-probe controller, whichever is
strongest/characterized) as reusable `arc_solver_kit` scaffolding + measure leave-one-game cross-game
transfer HONESTLY (a transfer null is a valid characterized result).

### PHASE A6 — SUBMITTED_AGENT_CONFIG INTEGRATION GATE (exp4731; carried fwd, SKIPPED in .434)
Integrate the strongest banked .435 change into SUBMITTED_AGENT_CONFIG (parity-hard-gated, no regression).
If all levers null, emit the honest-null markers (the TAUTOLOGY carve-out reads them).

### PHASE B2 — LEVER-EXERCISE-EVIDENCE GUARD (exp4732; operator MANDATORY 2026-06-25; carried fwd) — INFRA slot 2
Mechanize the silent-dead-code catch: an `adversarial_verify` check that flags a generation/exploration-
lever artifact declaring a mechanism exercised while its exercise evidence is degenerate (zero injections,
empty/unchanged pool, wrong-shape tensor, **byte-identical arms** — the .434 A4 signature). Pinned by a new
test. Do NOT weaken any existing check.

### PHASE C — KV260 HARDWARE CONTINUITY (exp4733; carried fwd, SKIPPED in .434)
SSH-reachability precondition (NEVER host SD-card device nodes) + on-board Ising latency transcript toward
terminal state. Honest `blocked_kv260_ssh_unreachable` if unreachable.

### PHASE D — SOTA INGESTION: the .436 fallback (exp4734)
Map the NEXT frontier beyond active-probe: the **epistemic-object-model MCTS probe planner**
(arXiv:2210.13455 + arXiv:2601.06604) + the **factored interaction/causal probe bank** (arXiv:2511.02225 +
arXiv:2511.14262), bounded by the object-world-model breakage falsifier (arXiv:2511.06136). Low-concurrency
WebSearch/WebFetch (NOT `/deep-research`). Real arXiv IDs for every claim.

### PHASE E — CAPSTONE (exp4735)
Aggregate the scorecard + the HEADLINE DECISION: did .435 cross the bridge? Did **A1** (validly-tested
online driver) prove non-degenerate arms AND beat frozen by +0.05 / deepen to L2? Did **A2** (active-probe)
reach a NEW level with the no-probe ablation failing? Did **B1** find the .434 A4 was a silent-bug that must
reopen? Did **A3** bank +1 (63→64)? Skip `flagged_adversarial`/control-failed/ablation-missing artifacts.
Confirm `verifier_is_oracle:false` + `solve_provenance` on every solve. Re-affirm G1–G4 `paper_ready`.

---

## 5. Dependency graph

```
exp4724 (phase0: archive/activate + resolve poison pre-test)
   +-> exp4725 (B1 silent-bug audit — runs early; adjudicates .434 A4 -> reopen list)
          +-(informs)-> exp4726 (A1 online driver — first valid test, non-degeneracy-gated)
   exp4726 (A1) --+
   exp4727 (A2 active-probe) --+
   exp4728 (A3 level-up self-play + verifier train) --+
                                                      +-> exp4729 (A4 readiness; reads A1/A2 config)
                                                      +-> exp4730 (A5 persist; reads A1/A2/A3)
                                                      +-> exp4731 (A6 integration; reads A1/A2)
   exp4732 (B2 exercise-evidence guard — mechanizes B1's catch)
   exp4733 (C KV260)        exp4734 (D SOTA ingestion)
   all --> exp4735 (E capstone .435)
```

Structured `gated_on` is intentionally NOT used between A1 and B1 — A1's reopen is self-justified by the
recorded .434 A4 TAUTOLOGY flag (a fact in the capstone), and B1 provides the formal confirmation in the
same milestone; gating A1 on B1 would serialize the two biggest tasks unnecessarily under the deadline.

---

## 6. Hardware requirements

| Phase | Hardware | Notes |
|---|---|---|
| A1, A2 | iGPU (Radeon 890M) for the frozen Qwen3.5-9B-MTP generator; RTX 3090 for the offline online-CNN training arms | Generator NEVER on the 3090s (project_arc_live_generator). Free port + `/props`-verify Qwen (the port-8919 gemma-squat confound). |
| A3, A4, A5, A6 | CPU (offline arcade + cached-candidate scoring) | `verifier_ensemble_against_cached_candidates` / aggregation substrates. |
| B1, B2, D, E, phase0 | CPU | aggregation / lint / literature synthesis. |
| C | KV260 over SSH (`ssh kria`) | SSH-reachability precondition; NEVER host SD-card. Honest blocked-record if unreachable. |

---

## 7. SOTA models (CLAUDE.md mandate)

The live ARC generator is **FROZEN** for the sprint: `unsloth/Qwen3.5-9B-MTP-GGUF` on the iGPU (MTP + q8 KV
+ `n_predict>=2048` + `/no_think`), per `project_arc_live_generator`. A1/A2 arms that induce/strategize via
the LLM declare `live_llm_inference` and construct the proposer on a FREE port + `/props`-verify Qwen. The
mandated SOTA GGUFs (`Qwen3.6-35B-A3B`, `gemma-4-31B-it`, `gemma-4-26B-A4B-it`) remain the headline-eligible
models for any non-ARC-generator LLM work; none is needed this milestone (ARC-only).

---

## 8. What this milestone explicitly RETIRES / does NOT re-propose

- The .434 A2 **static** present-winner ranker AS-BUILT (`present_winner_not_separable_from_distractors`) —
  static selection is exhausted; A2 (.435) changes the mechanism to **active probing** (act to disambiguate).
- The .434 A1 perception-grounded single-frame structural-alignment L2 goal (`alignment_under_determined`) —
  not re-proposed; the L2-goal-induction lineage is paused pending a fundamentally new grounding signal.
- The .433 single-exemplar and reinduction L2-goal fixes (already nulled, do not re-propose).
- Goal-free Go-Explore as a standalone L2 solver (a goal-DIRECTED L2 cannot be reached goal-free).
- Re-running the .433/.434 online-driver attempts **verbatim** — A1 (.435) is the FIRST *valid* test,
  gated on proven per-arm non-degeneracy (the prior attempts ran dead code).
- Any readiness claim that cites the replay-package level count as "the leaderboard score."

## 9. Compliance checklist (ARC sprint forcing functions)

- [x] Majority ARC: 6 ARC phases (A1–A6) fill 100% of the non-reserved slots.
- [x] ARC Level-Up Attempt Guarantee: ≥1 BANK attempt — A3 (self-play bank) + A1 (L2 probe) + A2 (new level).
- [x] Self-play every milestone: A3 trains + checkpoints the learned verifier.
- [x] 2 reserved infra: B1 (audit), B2 (guard) — both operator-MANDATORY, carried forward.
- [x] 1 per-board hardware: C (KV260).
- [x] 1 SOTA-ingestion: D.
- [x] Capstone: E.
- [x] All experiments `agent_type: codex` / `gpt-5.5`; planner + retro stay Claude Opus.
- [x] `verifier_is_oracle:false` on every value claim; `solve_provenance` on every ARC solve.
- [x] Submission operator-only; FoVer 0.9131 / G1–G4 `paper_ready` frozen.
</content>
