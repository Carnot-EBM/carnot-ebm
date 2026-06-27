# Research Roadmap — Milestone 2026.06.448

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-27.
**Milestone doc:** this file.
**Theme:** **Fork the generation wall: GUIDANCE-gap vs. world-model-INDUCER ceiling.**
The .447 diagnostic settled *where* the L1-first-contact wall is — **NEVER_ENUMERATED
dominant (9/10 held-out games)**: the live proposer enumerates hundreds-to-thousands of
candidates but the winning multi-step prefix is **never assembled**. The .448 headline asks
the one question the .447 finding leaves open and that two *retired* coverage levers force:
**why** is it never assembled — because the search has no GUIDING signal to assemble the
sufficient primitives (a planner gap, buildable), or because our air-gapped weak inducer
cannot induce an accurate-enough world model to plan through (a structural ceiling)? It is
the genuinely-untried direction (induce → `plan_in_model`, the Family-B SOTA path) measured
as a decisive **joint fork** — engine held-out accuracy × coverage migration — on the exact
.447 harness.

This milestone runs inside the **ARC-AGI-3 Submission Sprint** (CLAUDE.md forcing function,
active through **2026-06-30** — the deadline is ~3 days out). It therefore also drives the
realistic scored levers: a level-up bank attempt, self-play checkpointing, a *genuinely live*
held-out first-win readiness measurement (the 6/30 go/no-go signal), and final
submission-package hardening.

---

## 1. What the previous milestone (.447) proved

Read from the capstone (exp4859) and its audited upstreams:

| Phase | Result | Reading |
|---|---|---|
| **A1 — candidate-generation coverage diagnostic (HEADLINE)** | **NEVER_ENUMERATED dominant: 9/10 held-out games.** The live generic first-contact proposer enumerates 488–4766 candidates per game but the winning L1 prefix (len 4–33) is **never in the pool** (`matched_winning_prefix_len` 1–3, then the search diverges). The 1 COVERED game (lp85) and the tu93 positive control confirm the measurement is real. B1 (exp4855) confirmed it genuine: proposer was **blind** to the banked answer (no tautology), positive control genuinely COVERED, buckets match the per-game numbers (`b1_trusted: true`). | **The wall is GENERATION, and specifically the winning multi-step sequence is never assembled.** No ranker/selector can fix a pool that lacks the winner — the 17 prior ranking/selection/perception nulls are explained. `next_448_pivot: generation_expressibility_program_synthesis`. |
| **A2 — level-up attempt (scored lever)** | **NO bank.** Target s5i5 (rotated off .446's ka59); residual `needs_per_game_RE` (L2 marker-coverage delta not adaptered). `reproducible_total_levels` stays **65**. | The generation wall again: no new winning prefix found. |
| **A3 — self-play (every milestone)** | **PASS.** Learned verifier checkpoint refreshed, reproduction gate passed (`success_self_play_checkpoint_refreshed`). | The self-learning loop (train + checkpoint the learned verifier, FR-11) keeps working. |
| **A4 — held-out first-win readiness (deadline signal)** | **FLAT NULL at 0.04** (= 1/25), genuine no-improvement (positive control passed). This time a fresh number was produced. | Held-out generic first-win is pinned at chance-ish; the generation wall caps it. |
| **B2 — submission package** | **READY.** `vram_estimate_gb: 15.146` (< 16 GB Kaggle), package builds, operator-only. | The frozen Qwen3.5-9B-MTP stack packages for submission; the operator can submit. |
| **C — KV260 hardware** | **REACHABLE / graduated terminal.** SSH-only continuity (`success_kv260_continuity_ok`). | Keep in the per-milestone continuity rotation. |
| **D — SOTA ingestion (.448 frontier)** | Mapped 3 generation-expressibility tracks: DreamCoder/LILO library learning (2006.08381+2310.19791), neural-guided ARC DSL program search (2411.17708+2507.14172+2507.15877), COMET executable-world-model MCTS (2606.14418+2601.06604+2605.05138). | The handoff assumed "expressibility" = vocabulary expansion. Two retirements (below) refine that to GUIDANCE. |

**The reconciliation that defines .448 (load-bearing).** The .447 NEVER_ENUMERATED finding
*appears* to say "the primitive vocabulary is too small." But two levers were **already
RETIRED with empirical nulls** that say the opposite (ops/known-issues.md, 2026-06-23):

- **Macro-action vocabulary induction** — `complete: macro_horizon_collapse_empirical_null_guidance_not_depth`.
  Macros multiply branching (24 vs 4 candidates) without a guiding signal → strictly worse;
  on some games macros *hurt*. "The 0.04 wall is generation-**GUIDANCE**, not depth."
- **Click-heatmap-as-generator** — `complete: click_heatmap_generator_premise_falsified_guidance_not_coverage`.
  On 4,097 human effective clicks, **99.1% land ≤2px of an object centroid**; the centroid
  enumerator already covers what works. ARC click games are OBJECT-level ("which object, in
  which order"), not "where-precisely" → a coverage problem it is **not**.

**Synthesis:** the primitive vocabulary is already *sufficient*; NEVER_ENUMERATED means the
right sequence is never **ASSEMBLED**. The wall is **GUIDANCE / assembly**, not coverage.
Therefore .448 must **NOT** re-propose any coverage/vocabulary lever (macro / option-framework
/ off-centroid click generator are all retired). It must attack assembly.

---

## 2. The three biggest gaps (current state vs. north star)

North star (ops/north-star.md §0): a **live agent that discovers how to solve unseen games at
submission time**, accurately and efficiently. Gaps:

1. **ASSEMBLY / GUIDANCE (the binding gap, now one level deeper than .447).** The sufficient
   primitives are never assembled into the winner. The open fork: is this closable by a
   **guided planner** over those primitives (buildable — MCTS / world-model lookahead /
   neural-guided program search), or is it a structural **world-model-INDUCER ceiling** (our
   air-gapped weak Qwen3.5-9B-MTP cannot induce an accurate-enough forward model to plan
   through — the free-form engine nulled at 0.12 held-out accuracy)? **This fork is the gate
   that tells the whole program whether to build a planner or escalate the inducer.** We have
   never measured it.

2. **DEADLINE READINESS (the scored gap).** The package is ready and the operator can submit,
   but the only generalization signal — held-out first-win — is at the floor (0.04). The
   operator needs a *fresh live* number each milestone to make the 6/30 go/no-go call.

3. **CONTINUOUS SELF-LEARNING (the durable gap).** The learned verifier checkpoint improves
   across self-play runs (FR-11 relay), but its value is bounded by the generation/assembly
   wall — it can only route/rank a pool. Self-play must keep banking the verifier (Tier-3
   predictive verification) while the fork diagnosis reframes what the verifier should do next.

---

## 3. The decision: fork the wall with the genuinely-untried SOTA mechanism

> **Phase-Prototype + Empirical-Validation + Adversarial-Check discipline (CLAUDE.md):** a
> prototype must come with a measurable pass/fail and a hostile review *before* scaling.
> Applied here: before building an 18th lever (or escalating the inducer), **measure** which
> side of the fork the wall is on, with a positive control and an adversarial audit guarding
> the obvious tautology (feeding the banked answer into the planner).

**A1 (headline) is a decisive fork-probe run through a real mechanism.** For the held-out
games the .447 diagnostic found NEVER_ENUMERATED, it runs the **genuinely-untried** induce →
`plan_in_model` path (the Family-B SOTA approach, arXiv:2605.05138 — induce an executable
world model, then plan in it) **blind to the banked answer**, and measures, per game, the
**joint** result:

| Joint result | Meaning | What it implies for .449+ |
|---|---|---|
| **engine held-out accuracy HIGH + coverage migration NEVER_ENUMERATED → COVERED** | a decent induced model + planning assembles the winner the bare search never did | the wall was **GUIDANCE**; build out guided planning (MCTS / neural-guided program search) — buildable |
| **engine accuracy HIGH + NO migration** | the model is good but planning still can't assemble the winner | the **planner** is the gap → a stronger search/planner is .449 |
| **engine accuracy LOW + NO migration** | the induced world model is too inaccurate to plan through (the likely case — free-form engine ≈ 0.12) | the **INDUCER** is the structural ceiling → escalate to the operator (the weak-9B world-model wall the ledger names) |

Positive control: an adaptered/accurate-model game (tu93) must show HIGH engine accuracy +
migration — else a global "no migration" is a harness artifact, not a finding.

**Why this is not a doomed rerun / not a retired lever.**
- It is **NOT** macro-action vocabulary induction and **NOT** click-heatmap generation (both
  retired coverage levers) — it changes nothing about the primitive vocabulary; it tests
  *assembly* via planning.
- It is **NOT** the free-form-engine accuracy null (exp `e3.load_engine`, heldout 0.12): that
  measured engine accuracy as the deliverable. A1's deliverable is the **joint
  accuracy × coverage-migration fork on the exp4851 harness** — a different measurement (does
  planning put the winner into the pool?) that *attributes* the result to inducer vs planner.
- It is **NOT** the .447 A1 (bare-search coverage): A1 measures **planned** coverage
  (induce → plan), which no prior experiment did, jointly with engine accuracy.
- It is the exact direction the two retirements **redirect to** (the GUIDANCE class) and the
  SOTA path the levers ledger names as the structural gap.

---

## 4. Architecture / data flow for the A1 fork-probe

```
  banked offline solves (ops/arc_solve_registry.yaml, 25 games, 65 levels)
        │   winning L1 prefix  =  GROUND TRUTH  (held out from induction + planning)
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  for each NEVER_ENUMERATED held-out game (from exp4851), run BLIND:    │
  │  e3.load_engine  →  induce an executable world model from the agent's  │
  │  OWN cold-start transitions  →  plan_in_model  (Family-B SOTA path,    │
  │  live-path-reachable: arc_competition_agent._induce_and_plan)          │
  └──────────────────────────────────────────────────────────────────────┘
        │  (a) induced world-model HELD-OUT transition accuracy
        │  (b) planned candidate pool  →  COVERAGE CHECK vs banked winner
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  JOINT FORK:  accuracy × {COVERED / ENUMERATED-BUT-LOST / NEVER}       │
  │  HIGH+migration → GUIDANCE wall;  HIGH+no-migration → planner gap;     │
  │  LOW+no-migration → INDUCER ceiling.  Positive control: tu93 must be   │
  │  HIGH accuracy + COVERED (else a global null is a harness artifact).   │
  └──────────────────────────────────────────────────────────────────────┘
        │  per-game (accuracy, bucket) + the fork verdict
        ▼
  B1 adversarial audit: planner was BLIND to the banked answer (no tautology),
  on REAL games, positive control genuinely migrated, the fork verdict matches
  the per-game numbers.  →  .449 direction
```

The module must be **live-path-reachable** (importable by the live agent;
`arc_orphan_solver_lint` passes) — the natural home is an instrumentation harness that calls
the *existing* `arc_competition_agent` `e3.load_engine` / `plan_in_model` path, **not** a
parallel solver (CLAUDE.md "ARC Live-Path Reachability Discipline").

---

## 5. Phase plan (10 tasks, conductor execution order)

| # | id | Phase / track | What | Substrate | max_turns |
|---|---|---|---|---|---|
| 0 | exp4860 | TRANSITION | archive .447 → activate .448; record close-state (A1 NEVER_ENUMERATED dominant + b1_trusted; macro-vocab & click-heatmap RETIRED; energy concluded; exploration-prior closed; `reproducible_total_levels=65`); resolve any poison pre-test | aggregation | 40 |
| 1 | exp4861 | **A1 — ARC north star, HEADLINE** | **Generation-wall fork probe**: induce → `plan_in_model` (Family-B SOTA) on the NEVER_ENUMERATED held-out games; joint (engine held-out accuracy × coverage migration) ⇒ GUIDANCE-gap vs INDUCER-ceiling; positive control tu93 | live_llm_inference | 200 |
| 2 | exp4862 | A2 — ARC north star, **Level-Up Attempt Guarantee** | bank ≥1 NEW reproducible level on a ROTATED target (away from .447's s5i5 / .446's ka59) | live_llm_inference | 160 |
| 3 | exp4863 | A3 — ARC north star, **self-play / continuous self-learning** | standing `arc_loop_solve` loop: verifier-routed solve → reproduction gate → **train + checkpoint** the learned verifier (FR-11) | live_llm_inference | 140 |
| 4 | exp4864 | A4 — ARC north star, **deadline lane** | **GENUINELY LIVE** held-out first-win readiness (checkpoint/resume, wall-clock-safe); the 6/30 go/no-go signal | live_llm_inference | 120 |
| 5 | exp4865 | B1 — INFRA slot 1, **adversarial check** | audit A1: planner was BLIND to the banked answer (no tautology), real games, positive control genuinely migrated, the fork verdict matches the per-game accuracy×bucket numbers | aggregation | 100 |
| 6 | exp4866 | B2 — INFRA slot 2, **deadline** | submission-package FINAL harden + operator submission checklist (NEVER submits) | aggregation | 100 |
| 7 | exp4867 | C — HARDWARE | KV260 SSH-only continuity; ALWAYS write the deliverable (blocked artifact is correct if offline) | hardware_smoke | 60 |
| 8 | exp4868 | D — SOTA ingestion (.449 frontier) | given A1's fork verdict, ingest the matching SOTA (GUIDANCE → neural-guided planning / MCTS / world-model induction quality; INDUCER-ceiling → stronger/decentralized inducer options); real arXiv IDs only | aggregation | 100 |
| 9 | exp4869 | E — CAPSTONE | aggregate the scorecard: A1 fork verdict (B1-trusted), level-up bank, self-play, fresh live readiness, submission state, hardware, .449 handoff | aggregation | 120 |

**Dependency graph.** `exp4860` → all. `exp4865` (B1) audits `exp4861` (A1). `exp4868` (D)
reads A1's fork verdict to aim the ingestion. `exp4869` (E) aggregates all. The A-phase tasks
(A1–A4) are independent and run in conductor order; no A-task gates another A-task (avoids
same-milestone dependency stalls).

**ARC-sprint compliance.** Majority-ARC: of the non-reserved slots, all 4 are ARC north-star
(A1–A4). Reserved (per the sprint forcing function): 2 infra (B1, B2), 1 hardware-continuity
(C), 1 SOTA-ingestion (D), 1 capstone (E), 1 transition (phase 0). **Level-Up Attempt
Guarantee:** A2 (≥1 bank attempt, rotated target). **Self-play every milestone:** A3.
**Continuous self-learning:** A3 trains + checkpoints the learned verifier (FR-11 / Tier-3).

---

## 6. Hardware & model requirements

- **ARC live tasks (A1–A4)** use the **FROZEN sprint stack** (CLAUDE.md ARC Submission Sprint
  Forcing Function, operator-fixed 2026-06-19): generator = **Qwen3.5-9B-MTP** (5.9 GB Q4,
  Apache) on the **iGPU (Radeon 890M)** — **NEVER the 3090s** — with MTP + q8 KV +
  `n_predict>=2048` + `/no_think`. This freeze supersedes the general SOTA-GGUF mandate for
  ARC sprint work (a 35B GGUF would violate the Kaggle ~16 GB constraint and the freeze). A1
  invokes the LLM exactly where `e3.load_engine` / `plan_in_model` already invoke it; it must
  checkpoint/resume per game and honor a soft elapsed budget (the 2026-06-25 wall-clock fix)
  so it fits the codex wall-clock cap.
- **B/C/D/E tasks** invoke no LLM (aggregation / SSH smoke / web-sweep), so the SOTA-GGUF
  mandate does not apply; their substrate is declared honestly
  (`aggregation_from_upstream_artifacts` / `hardware_smoke`).
- **KV260** (C): reachable via `ssh kria`; SSH-only (host SD-card device nodes permanently
  retired). Graduated terminal — kept in the continuity rotation.
- **D (SOTA ingestion)** uses the RELIABLE channel only (`sweep_clusters.py` /
  `sweep_semscholar.py` + low-concurrency `WebSearch`/`WebFetch`); `/deep-research` is banned
  from the autonomous loop.

---

## 7. Disciplines honored

- **ARC-AGI-3 IS a Live Hidden-Game Discovery Agent** — A1 measures the LIVE induce→plan
  discovery capability; it is not a per-game trained weight. The deliverable is the reusable
  fork-probe + the redirected program.
- **ARC Live-Path Reachability** — A1 calls the live `arc_competition_agent`
  `e3.load_engine` / `plan_in_model` path; `arc_orphan_solver_lint` must pass.
- **ARC Solve Reproducibility + Solver-Reuse** — A2/A3 gate every solve on
  `arc_solver_kit.reproduce`; only reproduced levels count; registry updated.
- **`solve_provenance`** — declared on every ARC task: A2/A3 = `live_agent_self_discovery`;
  A1/A4 = `development_proxy` (A1 is an offline fork-measurement, not a banked live solve; A4
  is a held-out proxy — declared honestly, no false live claim).
- **Failed-Experiment Rerun Discipline** — A1 carries a `prior_failures` block citing the
  .447 diagnostic (exp4851) and the free-form-engine null, each with what is different and
  `retire_if_same_verdict: true`. The two retired coverage levers (macro-vocab, click-heatmap)
  are explicitly NOT re-proposed.
- **Phase-Prototype + Adversarial-Check** — A1 has a positive control (tu93) + the B1 hostile
  audit guarding the tautology trap (answer not fed to the planner).
- **Pre-Launch Preconditions** — every compute task opens with a PRECONDITIONS block (arcade
  reachable, iGPU-not-3090, frames/env present) → `blocked_*` on miss, never fabricate.
- **Verdict Terminal-Prefix** — every `honest_verdict` starts with
  `complete_/success_/passed_/shipped_`.
- **Principle-Annotated Artifact Fields** — every REQUIRED ARTIFACT FIELD and gate carries a
  one-line `principle:`.
- **Circularity / Oracle-Distinctness** — A1 declares `verifier_is_oracle: true` (the
  reproduction gate that defines "the winner" is the executable oracle); the fork-probe is a
  grounding measurement, not a moat claim.
- **Codex-Default v2 / ARC sprint routing** — all experiment tasks `agent_type: codex` +
  `model: gpt-5.5`; planner/retro stay on Claude Opus 4.8 (operator's deliberate sprint
  quality choice).
- **Operator-Only External Publication** — B2 prepares the package + checklist; it NEVER
  submits (no submit credentials).
- **Never remove existing content** — registry/docs are appended, not replaced.

---

## 8. Expected outcomes & what each tells us

- **A1 INDUCER-ceiling (LOW engine accuracy + no migration — the likely case)** → the binding
  wall is the air-gapped weak-9B world-model inducer (the SOTA gap). .449 escalates to the
  operator: a stronger/decentralized inducer, or accept the ceiling and re-scope. The sharpest
  convergence-forcing result.
- **A1 GUIDANCE-gap (HIGH accuracy + migration)** → planning over sufficient primitives closes
  the wall; .449 builds out guided planning (MCTS / neural-guided program search). A real
  forward path.
- **A1 planner-gap (HIGH accuracy + no migration)** → the inducer is fine but the planner is
  weak; .449 builds a stronger planner over the good model.
- **A2** banks +1 level (monotonic `reproducible_total_levels`) or records the rotation
  dead-end.
- **A4** delivers a *fresh live* held-out first-win number with CI — the operator's 6/30
  go/no-go signal.
- **A3** refreshes the learned verifier checkpoint (continuous self-learning).
- **B2** confirms the package is operator-submittable.

Whichever way the fork resolves, it converts the .447 NEVER_ENUMERATED finding into the
actionable next-level root cause (planner vs inducer) — the convergence the north star demands,
3 days before the deadline.
