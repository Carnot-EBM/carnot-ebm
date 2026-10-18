# Research Roadmap — Milestone 2026.06.447

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-27.
**Milestone doc:** this file.
**Theme:** **Stop building levers; diagnose the wall.** After ~17 consecutive
nulled candidate-generation/ranking/perception levers (.425–.446), and with the
.446 object-identity perception research bet landing a **genuine null**
(object identity is unrecoverable from the rendered grid even by shape/motion
tracking), the highest-leverage, least-wasteful, discipline-compliant move is
NOT an 18th lever. It is to **precisely measure where the winning first-contact
trajectory is lost in generation** — the single question every null has begged
but none has answered. This is the honest forward focus the standing operator
directive names: *"the L1-first-contact GENERATION wall (the actual multi-level
live-solve blocker)."*

This milestone runs inside the **ARC-AGI-3 Submission Sprint** (CLAUDE.md
forcing function, active through **2026-06-30** — ~3 days out). It therefore
also drives the realistic scored levers: a level-up bank attempt, self-play
checkpointing, a *genuinely live* held-out first-win readiness measurement (the
deadline go/no-go signal), and final submission-package hardening.

---

## 1. What the previous milestone (.446) proved

The .446 milestone ran operator-directed track 'c' (a research bet + a scored
lever, in parallel). Outcomes, read from the capstone (exp4849) and its audited
upstreams:

| Phase | Result | Reading |
|---|---|---|
| **A1 — object-identity perception probe (HEADLINE research bet)** | **GENUINE NULL.** A generic shape/connectivity/motion object tracker (recolor-invariant) beat a color-centroid baseline on only **1 of 3** real-frame games (r11l, the trivial 5-frame case). On the goal-grounding threshold lp85 (53 frames) and the tu93 positive control (94 frames) it did **not** recover stable cross-frame object identity. `goal_grounding_feasible: false`, `genuine_rendered_grid_null: true`. B1 confirmed it was genuinely exercised on REAL frames (not a synthetic non-test, not a baseline no-op). | **Object identity is not recoverable from the passive rendered grid.** The agent would need internal sprite state, which does not generalize to hidden games. **Grid-grounded goal-grounding via passive perception is retired.** A real, headline-worthy negative. |
| **A2 — level-up attempt (scored lever)** | **NO bank.** 19 games attempted (target ka59); 0 new reproducible levels. `reproducible_total_levels` stays **65**. | The candidate-generation wall again: no new winning prefix found. |
| **A4 — held-out first-win readiness (deadline signal)** | **FLAT NULL at 0.04** (= 1/25, the baseline), delta 0.0; positive control passed, parity green — **but `live_agent_ran: false`** (a cache resume / aggregation, not a fresh live run). | The held-out first-win rate is pinned at chance-ish. And we do not even have a *fresh live* number — .447 must produce one. |
| **A3 — self-play (every milestone)** | **PASS.** Learned verifier checkpoint refreshed (re86, L2), reproduction gate passed, `live_agent_self_discovery`. | The self-learning loop (train+checkpoint the learned verifier) keeps working. |
| **B2 — submission package** | **READY.** `vram_estimate_gb: 15.146` (< 16 GB Kaggle), package builds, operator-only. | The frozen Qwen3.5-9B-MTP stack packages for submission. The operator can submit. |
| **C — KV260 hardware** | **REACHABLE / graduated terminal.** SSH-only continuity. | Keep in the per-milestone continuity rotation. |
| **D — SOTA ingestion (.447 frontier)** | Mapped object-relational world-model / planning methods (comet_object_mcts, slot_mpc, loop_owm). **Caveat:** every method's own `fails_when` is *"the A1 tracker merges objects"* — exactly what A1 found. | The handoff assumed A1 would succeed. It did not. .447 cannot naively "consume the A1 object layer." |

**The cross-milestone pattern (the load-bearing fact).** Across the ~17 nulled
levers from .425 to .446 — graded goal-energy, per-level reinduction, verifier
router ranking, world-model trust gate, persistent action-effect memory,
env-adaptive resolve, approach dispatcher, value-head bridge, cheap value
routing, DAgger off-path, PoE-World expert trust, controllable novelty,
object-centric representation builder, online-warm action-effect controller,
energy-fitness QD, and now object-identity perception — **every single
`residual_dead_end` is the same sentence:** *"candidate generation remains the
residual bottleneck; the winning prefix is never proposed, so ranking/routing/
perception can only reorder a pool that does not contain the winner."*

The energy-as-ARC-lever program is **CONCLUDED** (2026-06-26 operator directive:
"do NOT re-propose energy stages"). The exploration-prior class is **CLOSED**.
Perception-from-grid is a **proven null**. What remains un-measured is the
**generation wall itself**: *does the live proposer ever even EXPRESS the
winning first-contact action sequence?*

---

## 2. The three biggest gaps (current state vs. north star)

The north star (ops/north-star.md §0): a **live agent that discovers how to
solve unseen games at submission time**, accurately and efficiently. Gaps:

1. **GENERATION (the binding gap).** Held-out generic first-win is **0.04 = 1/25**
   — the agent solves a fresh game's L1 essentially at chance. ~17 levers that
   *rank/route/perceive* an existing candidate pool have nulled because the pool
   does not contain the winner. **We have never measured why.** Is the winning
   prefix *enumerated-but-lost* (a ranking/pruning/budget gap a lever could still
   fix) or *never-enumerated* (an expressibility/vocabulary gap that no ranker can
   fix, and that demands program-synthesis over richer action primitives)? This
   diagnosis is the gate that tells the whole program where to spend its effort.

2. **DEADLINE READINESS (the scored gap).** The submission package is ready and
   the operator can submit, but the only generalization signal — held-out
   first-win — is (a) at the floor and (b) not freshly live-measured. The operator
   needs a *fresh live* number to make the 6/30 go/no-go call.

3. **CONTINUOUS SELF-LEARNING (the durable gap).** The learned verifier checkpoint
   improves across self-play runs (FR-11 relay), but its value is bounded by the
   generation wall — it can only route/rank a pool. Self-play must keep banking
   the verifier (Tier-3 predictive verification) while the generation diagnosis
   reframes what the verifier should learn to do next.

---

## 3. The decision: diagnose, don't build

> **Phase-Prototype + Empirical-Validation + Adversarial-Check discipline (CLAUDE.md):**
> a prototype must come with a measurable pass/fail and a hostile review *before*
> scaling. Applied here: before building an 18th generation lever, **measure** the
> generation wall, with a positive control and an adversarial audit guarding the
> obvious tautology (feeding the offline solver's answer back into the proposer).

**A1 (headline) is a DIAGNOSTIC, not a lever.** It instruments the **live**
generic first-contact proposer (the StepwiseExplorer path inside
`arc_competition_agent` / `arc_loop_solve`'s `OfflineSolver` proposal step) and,
for a set of games whose winning L1 prefix is **already banked offline as ground
truth**, measures whether the proposer — run *blind to the banked solution* —
ever puts the winning prefix into its candidate pool within the standard action
budget. It decomposes every failure into exactly one of three buckets:

| Bucket | Meaning | What it implies for .448+ |
|---|---|---|
| **COVERED** | winner enumerated AND reaches the L1 win within budget | the wall is **ranking/selection** → a ranking lever (or the learned verifier) can still help |
| **ENUMERATED-BUT-LOST** | winner enumerated but pruned / truncated by the search budget | the wall is **search depth / pruning** → widen the budget or fix the pruner |
| **NEVER-ENUMERATED** | the winner's action primitives are never proposed | the wall is **expressibility / vocabulary** → no ranker can fix it; need **program-synthesis over richer primitives** |

The hypothesis, given 17 ranking-lever nulls, is that **NEVER-ENUMERATED
dominates** — which would be a sharp, program-redirecting finding (it retires the
entire ranking-lever class for first-contact and points .448 at generation
expressibility). But the milestone is falsifiable either way: a COVERED-dominant
result would *revive* the ranking levers with a concrete target.

**Why this is not a doomed rerun.** No prior lever *measured proposer coverage of
the known-winning prefix*, decomposed, on held-out first-contact, with the
offline-banked winner as ground truth and the proposer blind to it. They all
*assumed* an answer and built a lever. This is the missing measurement, and it is
explicitly the standing operator-directed forward focus ("the L1-first-contact
GENERATION wall").

---

## 4. Architecture / data flow for the A1 diagnostic

```
  banked offline solves (ops/arc_solve_registry.yaml, 25 games, 65 levels)
        │   winning L1 prefix  =  GROUND TRUTH  (held out from the proposer)
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  LIVE generic first-contact proposer  (NO GameAdapter, NO injected   │
  │  answer) — StepwiseExplorer / OfflineSolver proposal step, run cold  │
  │  on each game's L1 within the standard action budget                 │
  └─────────────────────────────────────────────────────────────────────┘
        │  enumerated candidate pool  (the prefixes the proposer actually emits)
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  COVERAGE CHECK: is the winning L1 prefix (or any prefix that reaches │
  │  the L1 win) in the pool?  Decompose: COVERED / ENUMERATED-BUT-LOST / │
  │  NEVER-ENUMERATED.  Positive control: a game WITH an adapter must be  │
  │  COVERED (else a global null is a harness artifact).                  │
  └─────────────────────────────────────────────────────────────────────┘
        │  per-game bucket + the dominant bucket
        ▼
  B1 adversarial audit: proposer was BLIND to the banked solution (no
  tautology), measured on REAL games, positive control genuinely covered,
  buckets match the per-game numbers.  →  .448 direction
```

The module must be **live-path-reachable** (importable by the live agent;
`arc_orphan_solver_lint` passes) — a diagnostic the live agent cannot reach is
wasted effort (CLAUDE.md "ARC Live-Path Reachability Discipline"). The natural
home is an instrumentation hook on the existing `OfflineSolver` /
`StepwiseExplorer`, not a parallel solver.

---

## 5. Phase plan (10 tasks, conductor execution order)

| # | id | Phase / track | What | Substrate | max_turns |
|---|---|---|---|---|---|
| 0 | exp4850 | TRANSITION | archive .446 → activate .447; record close-state (A1 perception genuine-null, energy concluded, exploration-prior closed, `reproducible_total_levels=65`); resolve any poison pre-test | aggregation | 40 |
| 1 | exp4851 | **A1 — ARC north star, HEADLINE** | **Candidate-generation coverage diagnostic**: does the live proposer ever emit the known-winning L1 prefix? Decompose COVERED / ENUMERATED-BUT-LOST / NEVER-ENUMERATED, with a positive control | live_llm_inference | 200 |
| 2 | exp4852 | A2 — ARC north star, **Level-Up Attempt Guarantee** | bank ≥1 NEW reproducible level on a ROTATED target (away from .446's ka59) | live_llm_inference | 160 |
| 3 | exp4853 | A3 — ARC north star, **self-play / continuous self-learning** | standing `arc_loop_solve` loop: verifier-routed solve → reproduction gate → **train + checkpoint** the learned verifier (FR-11) | live_llm_inference | 140 |
| 4 | exp4854 | A4 — ARC north star, **deadline lane** | **GENUINELY LIVE** held-out first-win readiness (checkpoint/resume, now wall-clock-safe); the 6/30 go/no-go signal | live_llm_inference | 120 |
| 5 | exp4855 | B1 — INFRA slot 1, **adversarial check** | audit A1: proposer was BLIND to the banked answer (no tautology), real games, positive control covered, buckets match numbers | aggregation | 100 |
| 6 | exp4856 | B2 — INFRA slot 2, **deadline** | submission-package FINAL harden + operator submission checklist (NEVER submits) | aggregation | 100 |
| 7 | exp4857 | C — HARDWARE | KV260 SSH-only continuity; ALWAYS write the deliverable (blocked artifact is correct if offline) | hardware_smoke | 60 |
| 8 | exp4858 | D — SOTA ingestion (.448 frontier) | given A1's bucket finding, ingest **generation-expressibility** SOTA (program-synthesis-over-primitives / object-relational MCTS proposer) mapped onto the diagnosed gap; real arXiv IDs only | aggregation | 100 |
| 9 | exp4859 | E — CAPSTONE | aggregate the scorecard: A1 generation-wall verdict, level-up bank, self-play, fresh live readiness, submission state | aggregation | 120 |

**Dependency graph.** `exp4850` → all. `exp4855` (B1) audits `exp4851` (A1).
`exp4859` (E) aggregates all. The A-phase tasks (A1–A4) are independent of each
other and run in conductor order. No A-task gates another A-task (avoids
same-milestone dependency stalls).

**ARC-sprint compliance.** Majority-ARC: of the 4 non-reserved slots, all 4 are
ARC north-star (A1–A4). Reserved (per the sprint forcing function): 2 infra
(B1, B2), 1 hardware-continuity (C), 1 SOTA-ingestion (D), 1 capstone (E), 1
transition (phase 0). **Level-Up Attempt Guarantee:** A2 (≥1 bank attempt,
rotated target). **Self-play every milestone:** A3. **Continuous self-learning:**
A3 trains + checkpoints the learned verifier (FR-11 / Tier-3).

---

## 6. Hardware & model requirements

- **ARC live tasks (A1–A4)** use the **FROZEN sprint stack** (CLAUDE.md ARC
  Submission Sprint Forcing Function, operator-fixed 2026-06-19): generator =
  **Qwen3.5-9B-MTP** (5.9 GB Q4, Apache) on the **iGPU (Radeon 890M)** — **NEVER
  the 3090s** — with MTP + q8 KV + `n_predict>=2048` + `/no_think`. This freeze
  supersedes the general SOTA-GGUF mandate for ARC sprint work; substituting a
  35B GGUF would violate the Kaggle ~16 GB constraint and the freeze. The diagnostic
  (A1) is mostly CPU instrumentation of the proposer; the LLM is invoked only where
  the live proposer already invokes it.
- **B/C/D/E tasks** invoke no LLM (aggregation / SSH smoke / web-sweep), so the
  SOTA-GGUF mandate does not apply; their substrate is declared honestly
  (`aggregation_from_upstream_artifacts` / `hardware_smoke`).
- **KV260** (C): reachable via `ssh kria`; SSH-only (host SD-card device nodes
  permanently retired). Graduated terminal — kept in continuity rotation.
- **D (SOTA ingestion)** uses the RELIABLE channel only (`sweep_clusters.py` /
  `sweep_semscholar.py` + low-concurrency `WebSearch`/`WebFetch`); `/deep-research`
  is banned from the autonomous loop.

---

## 7. Disciplines honored

- **ARC-AGI-3 IS a Live Hidden-Game Discovery Agent** — A1 measures the LIVE
  proposer's discovery capability; it is not a per-game trained weight. The
  deliverable is the reusable diagnostic + the redirected program.
- **ARC Live-Path Reachability** — the A1 instrumentation hooks the live
  `OfflineSolver`/`StepwiseExplorer`; `arc_orphan_solver_lint` must pass.
- **ARC Solve Reproducibility + Solver-Reuse** — A2/A3 gate every solve on
  `arc_solver_kit.reproduce`; only reproduced levels count; registry updated.
- **`solve_provenance`** — declared on every ARC task: A2/A3 =
  `live_agent_self_discovery`; A1/A4 = `development_proxy` (A1 is an offline
  measurement; A4 is a held-out proxy — declared honestly, no false live claim).
- **Phase-Prototype + Adversarial-Check** — A1 has a positive control + the B1
  hostile audit guarding the tautology trap.
- **Pre-Launch Preconditions** — every compute task opens with a PRECONDITIONS
  block (arcade reachable, iGPU-not-3090, frames/env present) → `blocked_*` on
  miss, never fabricate.
- **Verdict Terminal-Prefix** — every `honest_verdict` starts with
  `complete_/success_/passed_/shipped_`.
- **Principle-Annotated Artifact Fields** — every REQUIRED ARTIFACT FIELD and gate
  carries a one-line `principle:`.
- **Circularity / Oracle-Distinctness** — A1 declares `verifier_is_oracle: true`
  (the reproduction gate that defines "the winner" is the executable oracle); the
  diagnostic is a grounding measurement, not a moat claim.
- **Codex-Default v2 / ARC sprint routing** — all experiment tasks
  `agent_type: codex` + `model: gpt-5.5`; planner/retro stay on Claude Opus 4.8
  (operator's deliberate sprint quality choice).
- **Operator-Only External Publication** — B2 prepares the package + checklist;
  it NEVER submits.
- **Never remove existing content** — registry/docs are appended, not replaced.

---

## 8. Expected outcomes & what each tells us

- **A1 NEVER-ENUMERATED-dominant** → the ranking-lever class is retired for
  first-contact; .448 pivots to **generation expressibility** (program synthesis
  over a richer action-primitive vocabulary). The strongest, most program-shaping
  result.
- **A1 ENUMERATED-BUT-LOST-dominant** → the wall is search budget/pruning; .448
  widens the budget / fixes the pruner (cheap, concrete).
- **A1 COVERED-dominant** → the winner IS in the pool; the 17 ranking nulls were a
  *ranking* failure, not a generation failure → revive the learned-verifier
  ranker with the coverage games as a proving set.
- **A2** banks +1 level (monotonic `reproducible_total_levels`) or records the
  rotation dead-end.
- **A4** delivers a *fresh live* held-out first-win number with CI — the operator's
  6/30 go/no-go signal.
- **A3** refreshes the learned verifier checkpoint (continuous self-learning).
- **B2** confirms the package is operator-submittable.

Either A1 outcome is a real, headline-worthy result that converts ~17 nulls into a
single sharp redirection — the convergence the north star demands.
