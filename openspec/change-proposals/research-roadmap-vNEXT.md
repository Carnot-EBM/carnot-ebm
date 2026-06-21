# Research Roadmap — Milestone 2026.06.419

**Theme: REACH DEEPER LEVELS via per-level GOAL RE-INDUCTION — the real score lever.**

Planned by: outer-loop planner (Claude Opus 4.8), 2026-06-21.
Sprint: ARC-AGI-3 submission sprint through 2026-06-30 (CLAUDE.md "ARC-AGI-3 Submission
Sprint Forcing Function"). Majority ARC; >=1 level-up attempt (ARC Level-Up Attempt
Guarantee); 2 reserved infra; 1 per-board hardware; 1 SOTA-ingestion. ALL experiments
`agent_type: codex` / `gpt-5.5`; planner + retro stay Claude Opus (operator quality choice).

---

## What .418 proved (and the pivot that makes .419)

`.418 attacked ACTION EFFICIENCY via navigation (collapse the RESET-replay tax). It NULLED,
and in nulling it located the real lever:

| `.418 phase | result | what it means for `.419 |
|---|---|---|
| A1 forward-walk navigation | NULL (flagged TAUTOLOGY: best==control==7761.5) | **Navigation/action-trimming is a DEAD score lever.** Reset-replay fallbacks dropped (1546->1534) and forward-walk hits rose (26->34), but TOTAL actions did not move, because the run terminates on a FIXED transition-budget (`len(transitions) >= explore_budget`), not on exhausting the reachable set. Nav efficiency cannot cut the action count. RETIRED. |
| A2 reach-deeper-levels | DIAGNOSIS (false-positive-flagged, content valid) | **The real barrier, and the load-bearing input for `.419.** See below. |
| A3 level-up attempt | SUCCESS: cd82 L2 offline-reproduced | `reproducible_total_levels` 48 -> 50. The Level-Up Guarantee was met. |
| A4 integration | NULL (`efficiency_moved: false`) | No lever raised `core_efficiency`; submitted config unchanged. |

**The score metric is per-level, not action-count.** The submission gate now measures the
AUTHORITATIVE leaderboard metric (`arc_agi.scorecard`): per-level
`min((human/agent)^2 * 100, 115)`, aggregated as an INDEX-WEIGHTED MEAN over ALL a game's
levels — **unsolved levels score 0 and drag it down; deeper levels carry more weight.**
`core_efficiency` baseline = **2.0074**. lp85 has 8 levels but the agent solves only L0->L1,
scoring ~2 of a possible ~100. **The score lever is REACHING DEEPER LEVELS, full stop.**

### The .418 A2 barrier diagnosis (exp4524) — the load-bearing finding

The A2 artifact was flagged `flagged_adversarial` on a FALSE-POSITIVE TAUTOLOGY
(`core_efficiency_baseline == core_efficiency_best == 2.0074` — equal because NO lever moved
efficiency, not because of a bug; same class as the identifier/seed TAUTOLOGY carve-out). The
`.418 capstone correctly skipped the flagged artifact and therefore LOST the diagnosis. The
content is valid and crisp. The L1->L2 barrier is **NOT depth** (raising the cap 45->90 just
exhausted budget, still L1). It is **per-level GOAL RE-INDUCTION**:

- `l2_win_condition_differs_from_l1: true` — L2 is a DIFFERENT goal than L1.
- `known_l2_transition_in_salience: null` — the L2 transition mechanic is NOT in the salience
  candidate set; the agent cannot even represent the L2 goal.
- `energy_signal_available: false`, `dsl_energy: null` — DSL/world-model induction does NOT
  engage after a level-up.
- A2's own `actionable_next_step`: *"force post-L1 DSL/goal-predicate induction and route the
  frontier toward the level-conditioned L2 predicate."*

### Three concrete causes, all confirmed in `arc_competition_agent.py`

1. `SUBMITTED_TARGET_LEVELS = 1` (line 65) — the agent STOPS after one level beyond start, so
   even if it could cross into L2 it halts at L1.
2. Induction escalates **once, only on stall-AND-not-won** (line 1148:
   `if stalled and not won and not self.induced`). After winning L1 (`won=True`) it never
   re-induces; the induce path is an unstuck-mechanism, not a cross-level mechanism.
3. The local gate sets `CARNOT_ARC_DISABLE_INDUCTION=1` (line 1176), so per-level efficiency
   is measured on the BARE explorer, which has no L2 mechanic at all.

**`.419 builds the fix the diagnosis prescribed.**

---

## The .419 thesis (one paragraph)

ARC-AGI-3 is, in the words of the official benchmark paper (arXiv:2603.24621), a test of
*"explore, infer goals, build internal models of environment dynamics, and plan."* Our agent
infers the L1 goal and stops. To reach deeper levels — the only thing that moves the score —
it must **RE-ACQUIRE the goal at every level boundary**: on a level-up, re-induce the
L_{n+1} win-predicate from the post-transition frames (which present a different goal), route
the frontier search toward states satisfying that re-induced predicate, and keep going past
`target_levels=1`. This is a per-level **refinement loop** (arXiv:2601.10904), and the Carnot
oracle-distinct slot is the world-model TRUST ENERGY used as a next-level-distance heuristic
(the 2026-06-20 operator strategic directive: energy that AUGMENTS the winners' search). The
primitive, once built, is captured as reusable scaffolding so the LIVE solver applies it to
games it has never seen.

```
                 .419 PER-LEVEL GOAL RE-INDUCTION LOOP
   clear L_n  ──►  detect level boundary (levels_completed bump)
                        │
                        ▼
              RE-INDUCE the L_{n+1} win-predicate           ◄── A1 (mechanism)
              from post-transition frames (Family-B
              executable world-model; goal differs)
                        │
                        ▼
              ROUTE the frontier toward states that         ◄── A2 (energy-trust routing;
              satisfy the re-induced L_{n+1} predicate           verifier_is_oracle:false,
              (world-model TRUST ENERGY = distance heuristic)     the oracle-distinct moat)
                        │
                        ▼
              raise target_levels; keep going ───► reach L_{n+1}
                        │                                (core_efficiency ↑ 2.0074)
                        ▼
              PERSIST the per-level predicate + routing     ◄── A5 (Tier-2 self-learning
              recipe to arc_solver_kit / registry               + ARC reuse discipline)
                        │
                        ▼
              WIRE the winning config into SUBMITTED        ◄── A4 (integration; the
              + re-measure core_efficiency end-to-end            HEADLINE metric)
```

---

## Phases (11 tasks)

### PHASE 0 — Transition (1 task, codex)
- **exp4532**: archive `.418 -> activate `.419; record the true `.418 close-state (nav/trim
  dead lever; A2 barrier diagnosis = per-level goal re-induction; A3 banked cd82 L2;
  `reproducible_total_levels=50`; `efficiency_moved=false`). Mechanical.

### PHASE A — ARC NORTH STAR (5 tasks; the score lever = `core_efficiency`>2.0074)
- **exp4533 (A1, HEADLINE)** — Build per-level GOAL RE-INDUCTION: detect the level boundary,
  re-induce the L_{n+1} predicate, route the frontier toward it, raise `target_levels`.
  Measure `core_efficiency` on lp85 + m0r0. Gate: a CORE game reaches L2
  (`core_efficiency` STRICTLY > 2.0074) at preserved CORE solves, OR a measured, actionable
  refinement of the barrier. Emits `model_specs`/`inference_substrate`/`efficiency_delta` +
  a null-delta methodology note (defuses the `.418 tautology false-positive that lost A2).
- **exp4534 (A2, HEADLINE; oracle-distinct moat)** — Energy-verifier next-level-distance
  routing: use the world-model TRUST ENERGY (`verifier_is_oracle: false`) to route the
  frontier toward predicted-deeper states. Test on lp85 + sp80 with a MATCHED no-energy
  control + FALSE_NEGATIVE_RISK guard. Self-contained (characterizes the energy signal even
  with no solve). Serves the 2026-06-20 energy-augmented-ARC directive.
- **exp4535 (A3, LEVEL-UP GUARANTEE)** — Bank +1 NEW reproducible level via the standing
  `arc_loop_solve` loop (deepen a shallow game: sp80/su15/cn04 L1->L2, or rotate a
  first-contact). Gate: `offline_reproduced=true AND reproduced_levels>=1`. Independent of
  A1/A2 so the guarantee holds even if the headline nulls.
- **exp4536 (A4, INTEGRATION + HEADLINE METRIC)** — Wire whatever RAISED `core_efficiency`
  (re-induction + raised `target_levels`) into `SUBMITTED_AGENT_CONFIG`; re-measure end-to-end
  on `core_efficiency`; keep `test_arc_submitted_agent_parity.py` green. Honest null if nothing
  raised it. `ready_for_operator_submit` (the task NEVER submits — operator-only).
- **exp4537 (A5, SELF-LEARNING + REUSE)** — Persist the per-level re-induction primitive +
  routing recipe to `arc_solver_kit` / `ops/arc_solve_registry.yaml` (Tier-2 constraint
  memory; the ARC Solve Reproducibility + Solver-Reuse Discipline), and measure CROSS-GAME
  TRANSFER (does the primitive reach a deeper level on a game it was not tuned on?). Degrades
  to "primitive registered + transfer-null characterized."

### PHASE B — Reserved infra (2 tasks)
- **exp4538 (B1, OVERDUE observability .363->.418)** — Wire the `.417 timing-detector repair
  into the RETRO timing-data path. The `.418 retro proved the repair reached the standalone
  detector but NOT the path feeding the retro's TIMING DATA block (still false-zeroed). Add a
  regression assert (injected count == on-disk in-window count) + `detector_gap_suspected`
  emission.
- **exp4539 (B2)** — Make capstone/aggregation ROBUST to known-false-positive flags: when an
  artifact's ONLY critical flag is a control==treatment null-delta TAUTOLOGY (the exact `.418
  pattern that LOST the A2 barrier diagnosis), the capstone/`summarize_artifact` must still
  read its diagnosis fields (with a corrigendum note) rather than excluding the whole
  artifact; ARC efficiency artifacts emit an explicit `efficiency_delta` + null-delta note.
  Asserting tests. (Artifact-emission + capstone-read robustness — NOT a relaxation of
  `adversarial_verify`'s fabrication detection.)

### PHASE C — Hardware continuity (1 task, operator_override)
- **exp4540 (C)** — Per-board audit: KV260 (SSH reachability ONLY, never host SD card),
  GateMate (USB detect), PolarFire (SSH). Honest `blocked_<board>_<reason>` per board.

### PHASE D — SOTA-ingestion (1 task, reserved; new track = GOAL ACQUISITION)
- **exp4541 (D)** — Ingest SOTA on per-level / intra-episode GOAL induction + goal-shift
  detection + program/world-model induction for ARC-AGI-3 (Family-B) + refinement-loop
  program synthesis. Map the strongest 3-5 onto the per-level re-induction headline; real
  arXiv IDs; flag the strongest for `.420. Note the `.418 SoRB/navigation thread is
  SUPERSEDED (nav is a dead score lever).

### PHASE E — Capstone (1 task)
- **exp4542 (E)** — The PER-LEVEL EFFICIENCY scorecard: did any lever RAISE `core_efficiency`
  above 2.0074 (reach a deeper level on a CORE game via re-induction)? Did the energy routing
  generalize? Did `reproducible_total_levels` grow (A3/A5)? Skip `flagged_adversarial` —
  BUT (per B2) do not lose a real diagnosis to a null-delta false-positive.

---

## Dependency graph

```
exp4532 (transition)
   └─► exp4533 (A1 re-induction mechanism, lp85+m0r0) ──┐
   └─► exp4534 (A2 energy-trust routing, lp85+sp80) ────┤   (A2 self-contained; does not hard-depend on A1)
   └─► exp4535 (A3 level-up bank, independent)          │
                                                        ▼
        exp4533/4534 ──────────────► exp4536 (A4 integration: wire winners, re-measure core_efficiency)
        exp4533 (primitive) ───────► exp4537 (A5 persist + cross-game transfer; degrades gracefully)
   └─► exp4538 (B1 retro timing-detector), exp4539 (B2 null-delta robustness)  [independent infra]
   └─► exp4540 (C hardware), exp4541 (D SOTA-ingestion)                        [independent]
        all A/B + registry ────────► exp4542 (E capstone .419 scorecard)
```

Graceful degradation (the `.409 lesson): A2 measures the energy signal even if A1 nulls; A4
honest-nulls if nothing raised efficiency; A5 persists the primitive and reports a transfer
null if no deeper level transfers; E aggregates what exists and notes gaps.

## Hardware requirements

- **None new.** ARC tasks run `inference_substrate: verifier_ensemble_against_cached_candidates`
  (offline arcade, all 25 games, zero quota; both RTX 3090s correctly idle). The induction
  tier, if invoked, uses the frozen live generator (Qwen3.5-9B-MTP on the iGPU, NEVER the
  3090s); the local gate disables induction (`CARNOT_ARC_DISABLE_INDUCTION=1`) for clean fast
  search measurement. PHASE C is `hardware_smoke` (SSH/USB board reachability only).

## Disciplines honored

- ARC-AGI-3 Submission Sprint (majority ARC; codex experiments; Opus planner/retro).
- ARC Level-Up Attempt Guarantee (A3 banks a new reproducible level; lint-checkable).
- ARC-AGI-3 Incremental-Progress Scoping (+1..+n levels on ONE game; no "FULL solve" task).
- ARC Solve Reproducibility + Solver-Reuse (A5 persists the primitive; only offline-reproduced
  levels count).
- Reserved infra (2: B1/B2), hardware continuity (1: C), SOTA-ingestion (1: D).
- Circularity / Oracle-Distinctness (A2 sets `verifier_is_oracle: false`).
- Adversarial Artifact Verification + Inference-Substrate Declaration + Verdict Terminal-Prefix
  + Principle-Annotated Artifact Fields + Pre-Launch Preconditions.
