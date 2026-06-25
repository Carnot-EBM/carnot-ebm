# Research Roadmap — Milestone 2026.06.434

**Status:** PROPOSED (outer-loop Claude Opus 4.8 planner, 2026-06-24)
**Theme:** **THE L1 WALL SPLIT IN TWO — perception is SOLVED (the winner is now PROPOSABLE),
SURFACING is the new bottleneck.** `.433 was the 9th-consecutive `bridge_crossed_for_solve=FALSE`
milestone, but its A1 perception diagnostic produced the most decisive new input in months:
**`perception_is_the_wall=true`** — an object-centric/relational representation makes the
winning L1 trajectory FULLY PROPOSABLE (coverage 1.0 vs order-1's 0.75), where eight prior
milestones of levers (all conditioning on order-1 perception at LOO=chance) could not. BUT the
winner sits at **rank 59/161 of 186** and the agent cannot SURFACE it (`generic_agent_reached_
level=0`, residual `offpath_calibration_insufficient`). `.434 attacks the surfacing layer (A2,
the verifier earns its place), BANKS the perception win as a multi-level solve (A1 HEADLINE:
lp85 L1->L2 via a perception-grounded structural-alignment goal — operator 2026-06-25 "yes"),
runs the CORRECTED goal-free online-action-learning DRIVER (A4 — operator 2026-06-24
leader-gap), and AUDITS the recent nulls for the SILENT BUGS the operator found (B1 — two of
the `.433 nulls tested DEAD CODE).
**Sprint:** ARC-AGI-3 Submission Sprint (CLAUDE.md, through **2026-06-30** — 6 days to the ARC
Prize Milestone #1 deadline, $25K).
**Predecessor:** 2026.06.433 (`research-roadmap.yaml`, capstone exp4710).

---

## 1. What the previous milestone (.433) proved

`.433 attacked the L1-first-contact wall with the two deepest unattacked root causes: A1
object-centric/relational PERCEPTION wired into the live PROPOSAL distribution (gated by a
decisive perception-vs-search diagnostic), and A2 AMORTIZED cross-game exploration + the
Go-Explore archive wired live. The capstone (exp4710) verdict: **`capability_grew_61_to_62`**
— A3 banked a level, but **`bridge_crossed_for_solve=FALSE` for the 9th consecutive milestone**,
and BOTH headline mechanisms nulled on live first-win (FLAT at 0.04 = 1/25 games).

| Task | Lever | Verdict | Outcome |
|---|---|---|---|
| A1 (exp4700) | Object-centric perception -> PROPOSAL distribution + perception-vs-search diagnostic | `complete: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient` | **NULL on the bank — but the DIAGNOSTIC RESOLVED THE 8-MILESTONE AMBIGUITY.** `perception_is_the_wall=true`: object-centric coverage **1.0** vs order-1 **0.75** (delta +0.25) — the winning L1 trajectory is now FULLY PROPOSABLE. The deployable arm still reached level 0 because the winner is at **rank 59, 161 of 186** candidates. Residual: `offpath_calibration_insufficient`. |
| A2 (exp4701) | Amortized cross-game exploration prior + Go-Explore archive wired live | `complete: amortized_prior_go_explore_no_coverage_gain_residual_logged` | **NULL — but TESTED DEAD CODE.** `coverage_delta=0.0`, `first_win_rate_delta=0.0`. The operator then found the Go-Explore archive `_frame_grid` returned a (1,64,64) 3-D array -> the archive was **SILENTLY DEAD** (fixed 2026-06-25). The amortized-exploration null is therefore UNTRUSTWORTHY. |
| A3 (exp4702) | Level-up self-play (rotation target) | `success` | **BANKED** +1 -> `reproducible_total_levels` 61->62. Learned verifier checkpointed. |
| A4 (exp4703) | Held-out first-win readiness lane (retargeted) | `complete: held_out_first_win_flat_no_leaderboard_change` | **FLAT** at 0.04. Null-delta markers now emitted (TAUTOLOGY carve-out satisfied). |
| online-action-learning (exp4710 arms) | Leader's online-CNN loop {frozen, online-scratch, online-warm, ...} | `online_action_learning_no_first_win_lift_null` (best_delta=+0.0000) | **NULL — but BUGGY.** Ran on a CNN dict-candidate silent bug (operator-confirmed). Untrustworthy. |
| Capstone (exp4710) | Scorecard + G1-G4 | `complete: capability_grew_61_to_62`, `paper_ready=True` | FoVer 0.9131 frozen; `bridge_crossed_for_solve=False`. |

### The decisive new input: the wall is TWO layers, and perception's is solved

For eight milestones the open question was *perception vs search/generation*. `.433's A1
diagnostic settled it: **perception IS (part of) the wall, and an object-centric representation
fixes the proposability half.** The winning L1 trajectory, invisible to order-1 frame features
(LOO=0.503=chance), becomes coverage-1.0 PROPOSABLE under object-centric/relational features.
The persisted primitive `object_centric_representation_builder_operator` (exp4704) ships this.

But proposability is not selection. The winner is PRESENT at rank 59 — the agent's value/
ranking calibration cannot surface a rank-59-then-rank-161 candidate across sequential steps
under budget. **This is exactly the slot the north-star verifier is built for** (router/pruner/
ranker, oracle-distinct): surface the present-but-buried winner. The prior selection nulls
(.425-.431) were on the WRONG (order-1) representation where the winner was absent; over the
object-centric coverage-1.0 pool a ranking lever finally has the winner to surface.

### The operator's two pre-stages + the silent-bug mandate

- **2026-06-25 ("yes" pre-stage): PERCEPTION-GROUNDED STRUCTURAL-ALIGNMENT L2 GOAL.** Convert
  the perception win into a banked multi-level solve: express lp85's L2 `is_level_complete` as
  a STRUCTURAL ALIGNMENT predicate over DETECTED objects (the A1 perception representation),
  computed from ONE live frame, wired into `level_up_reinduction`. -> `.434 A1 (HEADLINE).
- **2026-06-24 (leader-gap): GOAL-FREE ONLINE ACTION-LEARNING.** The StochasticGoose loop — an
  online frame-change CNN with a coordinate head that PROPOSES clicks + per-level reset,
  goal-free. -> `.434 A4 (CORRECTED — the `.433 attempt tested buggy code).
- **2026-06-25 silent-bug mandate:** two `.433 nulls tested DEAD CODE (Go-Explore (1,64,64);
  exp4710 CNN dict-candidate). "Audit the `.428-`.433 generation-lever nulls for other silent
  representation no-ops before trusting them." -> `.434 B1 (INFRA).

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **First-contact solve-rate is 0.04 (1/25) — the north-star metric is stuck.** `.433 proved
   the winner is PROPOSABLE under object-centric perception but not SURFACED. The gap is the
   ranking/calibration layer (A2): an oracle-distinct verifier that lifts the present winner
   from rank 59 to actionable top-k. This is the verifier earning its place (north-star §5).
2. **Multi-level depth is the second scored lever and it is goal-grounding-bound.** Every game
   stalls at L1->L2 because the induced goal predicate is degenerate. `.433's perception
   primitive gives the missing ingredient — a STRUCTURAL goal over detected objects (A1).
3. **Recent nulls may be silent-bug artifacts, not real limits.** Two `.433 nulls tested dead
   code. The loop's trust in its own negatives is at risk; B1 audits and re-validates.

---

## 3. Milestone architecture

```
                       L1 FIRST-CONTACT WALL  (first_win = 0.04, 9x not crossed)
                                     |
        .433 DIAGNOSTIC: perception_is_the_wall=TRUE -- the wall splits in two
                    /                                        \
   LAYER 1: PERCEPTION / PROPOSABILITY                 LAYER 2: SURFACING / RANKING
   object-centric coverage 1.0 (SOLVED)               winner present at rank 59 (OPEN)
                    |                                            |
        A1 (HEADLINE): bank it as depth              A2: surface the present winner
        lp85 L1->L2 via perception-grounded          off-path-calibrated oracle-distinct
        structural-alignment GOAL                    verifier/value ranker over the
        (operator 2026-06-25)                        object-centric coverage-1.0 pool
                                                     (verifier earns its place)
                    \                                        /
                     A4: goal-free online-action-learning DRIVER (corrected, bug-fixed)
                         coordinate-head-proposes-clicks; attacks BOTH layers
                         (operator 2026-06-24 leader-gap)
                                     |
   A3 self-play banks +1 (62->63) + trains verifier  |  A5 held-out first-win readiness
   A6 persist+transfer  |  A7 SUBMITTED_AGENT_CONFIG integration gate
                                     |
   B1 SILENT-BUG AUDIT of .428-.433 nulls   |   B2 adversarial_verify exercise-evidence guard
   C  KV260 hardware continuity   |   D  SOTA-ingestion (.435 fallback)   |   E  capstone v434
```

### Phases

- **Phase 0 (transition):** archive `.433 -> activate `.434; record the true close-state
  (capability 61->62, `perception_is_the_wall=true`, A2/online-action-learning nulls tested
  dead code, bridge not crossed 9th time).
- **Phase A (ARC NORTH STAR — majority of the milestone):**
  - **A1 (HEADLINE):** perception-grounded structural-alignment L2 goal -> lp85 L1->L2 bank.
  - **A2:** surface the present winner — off-path-calibrated oracle-distinct verifier/value
    ranker over the object-centric coverage-1.0 proposal pool (rank 59 -> top-k).
  - **A3:** level-up self-play (rotated clean game) + train/checkpoint the learned verifier.
  - **A4:** CORRECTED goal-free online-action-learning DRIVER (coordinate-head-proposes-clicks,
    bug-fixed Go-Explore + CNN; A/B {frozen, online-scratch, online-warm}).
  - **A5:** held-out first-win readiness lane (experiment_4605, bootstrap-CI, null-delta markers).
  - **A6:** persist the strongest `.434 primitive + cross-game transfer.
  - **A7:** SUBMITTED_AGENT_CONFIG integration gate (honest-null markers).
- **Phase B (INFRA — 2 reserved slots):**
  - **B1:** SILENT-BUG AUDIT of the `.428-`.433 generation-lever nulls (operator-mandated).
  - **B2:** adversarial_verify "lever-exercise-evidence" guard (mechanizes the silent-dead-code
    catch) + the perception-overclaim guard.
- **Phase C (HARDWARE):** KV260 SSH-reachability + latency-transcript continuity (north-star
  §3: KV260 is THE sovereignty story, drive to terminal then freeze).
- **Phase D (SOTA-INGESTION):** map the `.435 frontier (active-probe / hypothesis-driven
  world-model induction, arXiv:2506.01876 + 2309.08477; factored object-relational executable
  world model, arXiv:2511.02225/2410.08822).
- **Phase E (CAPSTONE):** scorecard + the HEADLINE DECISION (below) + G1-G4 re-affirm.

### The capstone HEADLINE DECISION (what E must adjudicate)

Did `.434 cross the offline->live bridge that nine milestones could not?
- Did **A1** bank lp85 L2 via a perception-grounded structural goal (offline-reproduced,
  `live_agent_self_discovery`, `goal_predicate_satisfiable=true`)? -> the perception win
  becomes a real multi-level SOLVE.
- Did **A2** surface the present-but-buried winner (precision-at-k up, the GENERIC agent
  reaches a NEW level with the no-surfacing ablation FAILING, offline-reproduced)? -> the
  verifier earns its place at the surfacing layer.
- Did **A4** (corrected online driver) beat frozen by >=+0.05 held-out first-win AND/OR deepen
  to L2? -> the leader's loop crosses the wall by demoting goal-induction.
- Did **B1** find that any `.428-`.433 null was a silent-bug artifact (a previously "closed"
  lever that must reopen)? -> trust correction.
- A3 banked +1 (62->63)? A5 held-out first-win readiness vs 0.04?
- Re-affirm G1-G4 `paper_ready` (FoVer 0.9131 frozen). Skip flagged/control-failed artifacts.
  Confirm `verifier_is_oracle:false` on every value claim; `solve_provenance` on every solve.

---

## 4. Dependency graph

```
exp4711 (phase0) ─► everything

exp4712 (A1 perception-grounded L2 goal) ─┐  (rides on the .433 object-centric primitive)
exp4713 (A2 surface present winner)       ├─► exp4717 (A6 persist) ─► exp4718 (A7 integration)
exp4714 (A3 self-play +1)                 │                                    │
exp4715 (A4 online driver, corrected)     │                                    ▼
exp4716 (A5 held-out readiness) ──────────┘                          exp4723 (E capstone)
exp4719 (B1 silent-bug audit) ─► informs whether A2-amortized/online nulls must reopen
exp4720 (B2 exercise-evidence guard)
exp4721 (C KV260)      exp4722 (D SOTA-ingestion)
```

A6/A7/E gate on the A1-A5 outcomes via structured `gated_on` where applicable (the capstone
reads all upstream artifacts; the integration gate reads the chosen submitted-config deltas).

---

## 5. Hardware requirements

| Task | Hardware | Note |
|---|---|---|
| A1, A2, A4 | iGPU (Radeon 890M) for Qwen3.5-9B-MTP GGUF | The FROZEN live generator (project_arc_live_generator). NEVER the 3090s. Free port + /props-verify (port-8919 gemma-squat confound). |
| A4 offline arms | RTX 3090 (CUDA) for CNN Adam training | Plus a CPU wall-clock measurement of an online step (Kaggle is CPU under a 12h/600-RPM cap — the #1 Kaggle-viability risk). |
| C | KV260 via `ssh kria` | SSH-reachability precondition (NEVER host `/dev/mmcblk*`). Latency transcript toward terminal state. |
| A3, A5, B1, B2, D, E | CPU only | Offline arcade / verifier-scoring / aggregation. |

---

## 6. Discipline compliance

- **ARC sprint forcing function:** majority ARC (A1-A7 = 7 of 13); >=1 level-up attempt that
  BANKS (A1 lp85 L2 + A3 rotated game = 2); self-play every milestone (A3); 2 reserved infra
  (B1/B2); 1 per-board hardware (C); 1 SOTA-ingestion (D). All experiments codex/gpt-5.5;
  planner/retro stay Claude Opus.
- **ARC Live-Path Reachability:** every A-task touches the LIVE modules (E3AgentPolicy /
  StepwiseExplorer / arc_value_learner / arc_llm_reinduction); `arc_orphan_solver_lint` +
  `test_arc_submitted_agent_parity` stay green. No parallel off-path solver.
- **solve_provenance:** every ARC solve task declares it; A1/A2/A4 target
  `live_agent_self_discovery`. Registry-precheck before any solve (no duplicate of a banked level).
- **verifier_is_oracle:false** on every value claim (the perception representation, the ranker,
  and the online CNN are all oracle-distinct from the executable reproduction win-check).
- **Circularity discipline:** A2's verifier is oracle-DISTINCT (a learned/energy ranker, not the
  win-check) — this is the gate-eligible, non-circular slot.
- **Failed-Experiment Rerun:** A1/A2/A3/A4 carry `prior_failures` with `retire_if_same_verdict:
  true`; routine continuations (phase0/A5/A6/A7/B2/C/D/E) carry `operator_override`.
- **Principle-annotated artifact fields** on every task; PRECONDITIONS step-0 on every
  compute-bound task; terminal-prefix `honest_verdict`.

---

## 7. RETIRED / do-NOT-re-propose

- The `.433 A1 deployable object-centric arm AS-BUILT (`offpath_calibration_insufficient`) —
  A2 changes the mechanism (a dedicated off-path-calibrated verifier/ranker, not the explorer's
  own value head).
- The `.430 single-exemplar L2-goal-fix (`single_exemplar_goal_insufficient`) — A1 uses a
  STRUCTURAL predicate over detected objects, not a flat exemplar grid.
- Goal-free Go-Explore deepening as a STANDALONE L2 solver (proto_goalfree_deepen null) — A1
  builds a satisfiable goal; A4 uses the (bug-fixed) archive only as a coverage component.
- Any A5 readiness claim that cites the replay-package level count as "the leaderboard score".
- Re-running the `.433 A2 amortized-exploration / online-action-learning nulls VERBATIM — they
  tested dead code; B1 must first confirm the fix, then A4 re-tests the corrected path.

6 days to the ARC Prize Milestone #1 deadline (2026-06-30). Submission operator-only.
