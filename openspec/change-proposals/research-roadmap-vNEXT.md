# Research Roadmap — Milestone 2026.05.315

**Depth-Over-Breadth: P0.1 Energy-Descent Premise, P0.2 Verifier Diversity,
Kona Solve-Rate Gate, and G2 Reproduction Closure**

**Status:** PROPOSED
**Planned:** 2026-05-30 (Opus 4.8 planner)
**Milestone:** 2026.05.315
**Experiment IDs:** exp3416, exp3312, exp3313, exp3417–exp3424
**Predecessor:** 2026.05.314 ("Latent Energy Spills, Neural Uncertainty Phase
Transitions, and Kona Global Reasoning")

---

## 1. Why this milestone exists — the convergence correction

The project runs ~30 milestones/day, has produced 3,400+ artifacts and 450+
milestones, yet `paper_ready` has been `False` throughout and the version
numbers (cross-corpus matrix v38, telemetry v39, repair panel vN, SOTA receipt
vN) climb in lockstep — the signature of **iteration without convergence**
(`ops/north-star.md`).

Milestone **.314 was that pattern**: NUP metric, latent-spills sensing,
telemetry aggregation v39, FR-11 stress vN, evidence matrix v38 — breadth churn
that re-measured the Phase-1 foundation while never running the load-bearing
links that decide the endgame. Two of its "perfect-score" artifacts
(exp3397 `duration_s=2.06` on a live 35B + AUROC=1.0; exp3405 accuracy=1.0,
no methodology) were caught by the adversarial-verify fabrication gate and are
quarantined.

The operator's response was the **Depth-Over-Breadth Forcing Function**
(CLAUDE.md, 2026-05-30) and the move of the planner to Opus 4.8 "precisely so
this prose discipline is followed reliably." **.315 is the first milestone
authored under that discipline.** The mandate:

> Until P0.1 (exp3312 / the Kona global-opt solve-rate gate) has a recorded
> verdict, the planner MUST reserve the majority of each milestone for the
> existential link tests (P0.1 energy-descent-vs-AR, P0.2 verifier-diversity,
> transpilation round-trip) and for **closing G2** (one independent reproducer
> of the FoVer 0.9131 headline). No `vN+1` re-measurement of an already-measured
> artifact unless it answers a NEW question.

These experiments are not new inventions — the operator **hand-authored** them
in `ops/known-issues.md` (P0.1 exp3312, P0.2 exp3313, the Kona correctness gate,
the ensemble-vs-injection test). They were queued at milestones .82/.83 (Zenil
stack) and .94–.97 (Phase-5 de-risking) and **never run** across 200+
milestones. .315 finally runs them.

---

## 2. What the predecessor (.314) actually proved

| Result | Status |
|---|---|
| Kona global-opt on hard Sudoku (exp3408): `init E=2104 → final E=10.05`, `solved=False`, 14.1s vs AR 212.2s | Honest, but the "15x speedup" is **fast-but-wrong-vs-slow** — meaningless until it SOLVES. Directly motivates exp3417. |
| EBM-CoT live benchmark (exp3397), NUP metric (exp3405) | **flagged_adversarial=true** — quarantined, NOT headline-eligible (2026-05-30 corrigendum). |
| Latent-spills / abductive-CSP / VGS / semantic-deficit (exp3406/3407/3409/3412) | Breadth probes; none moved the headline claim or a G-gate. |
| FR-11 stress / adversarial-robustness vN (exp3410/3414) | Re-measurement of an already-measured continual-learning capability — churn under the new rule. |

**Net:** .314 produced zero movement on the headline (FoVer 0.9131) and zero
movement on the publication gate (G1–G4). That is exactly the failure the
forcing function targets.

---

## 3. The three biggest gaps between current state and the PRD vision

1. **The Kona/foundation-model premise is untested.** The entire Phase-3
   endgame assumes energy-descent reasoning on continuous latents beats
   autoregressive token sampling. It has only ever been tested on toy 5×5
   puzzles (exp1222) and a downgraded BFS tie (exp1210) — never on a real task
   vs a real AR baseline. **→ P0.1 (exp3312).**

2. **The α_t grounding keystone may not hold at production k.** Self-correcting
   self-distillation only avoids collapse if the verifier ensemble has real
   diversity (small joint null space). exp1224 showed k=3 collapsing to
   effective k=1; the FoVer headline showed 3 of 4 verifiers contributing zero.
   If even a deliberately disjoint-kernel suite collapses, the self-improvement
   thesis needs a different foundation. **→ P0.2 (exp3313).**

3. **The paper cannot ship — G2 is the sole unmet gate.** G1/G3/G4 are met or
   near; G2 (independent reproduction) is the only real blocker, and the
   headline is *cheap, CPU-only, externally reproducible*. **→ exp3419** ships
   the turnkey clean-room reproducer and confirms in-CI recompute.

---

## 4. Milestone architecture

```
                    .315 — Depth-Over-Breadth
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   DEPTH BLOCK            HARDWARE              OPS / GATE
   (majority)            (light/opportunistic) (transition)
        │                     │                     │
  ┌─────┴──────┐        ┌─────┴─────┐         ┌─────┴─────┐
  │ exp3312 P0.1│       │exp3420 KV260│        │exp3416 archive│
  │ energy↓ vs AR│      │  → terminal │        │  .314→.315   │
  ├────────────┤        ├───────────┤         ├─────────────┤
  │ exp3313 P0.2│       │exp3421 Gate-│        │exp3423 G1-G4 │
  │ λ_min(Σ)    │       │ Mate root-  │  gated │ gate status  │
  │ diversity   │       │ cause diag  │  ────► │ (P0.1/G2     │
  ├────────────┤        ├───────────┤         │  verdicts)   │
  │ exp3417 Kona│       │exp3422 Polar│        ├─────────────┤
  │ solve-rate  │       │Fire opportun│  gated │exp3424       │
  │ correctness │       └───────────┘         │ capstone v315│
  ├────────────┤                              └─────────────┘
  │ exp3418     │
  │ ensemble vs │   ← architecturally-honest test of the thesis
  │ injection   │     on the corpus where the single KAN scored 0.475
  ├────────────┤
  │ exp3419 G2  │   ← turnkey clean-room reproducer; SOLE unmet gate
  │ reproduction│
  └────────────┘
```

**Task weighting (11 tasks):** 5 substantive depth/headline/gate tasks
(exp3312, 3313, 3417, 3418, 3419), 3 light hardware (KV260 terminal + 2
opportunistic), 3 ops (archive, gate-status, capstone). The majority of
*substantive* compute is the depth block — satisfying the forcing function.

---

## 5. Phase descriptions

### Phase A — The existential premise tests (depth)

- **exp3312 — P0.1 (CRITICAL, routed to Claude+Opus, `requires_claude`).** The
  single most important experiment in the project. Paired head-to-head:
  energy-descent reasoning (continuous-latent refinement on the Boltzmann-GPT
  substrate guided by verifier energy) vs an AR baseline of comparable
  parameter count, on a real GSM8K subset (n≥200), with McNemar/bootstrap
  significance. Routed to Claude+Opus because it meets all three `requires_claude`
  criteria: gemini has never trained the substrate to beat AR; 5+ files of
  choreography (boltzmann_gpt.py + continuous_ebm.py + verifier energy + corpus
  + AR baseline + stats); open-ended judgment on refinement schedule and compute
  parity that no deterministic gate substitutes for. **Either outcome is
  high-value** — validation greenlights Phase 3, refutation honestly retires the
  endgame.

- **exp3313 — P0.2 (CRITICAL, gemini).** Measures λ_min(Σ), participation-ratio
  effective-k, and per-verifier drop-one-out contribution on the broadest
  disjoint-kernel suite (structural / empirical / semantic / anti-vacuity /
  memory), on FoVer + an adversarial slice. Cheap (verifier scoring, reuses the
  FoVer corpus). **This is the milestone's continuous-self-learning experiment**
  (research-program.md requirement): the α_t grounding it tests is the precondition
  for the entire Tier-1→Tier-4 self-learning architecture.

- **exp3417 — Kona solve-rate correctness gate (CRITICAL, gemini).** Re-gates
  exp3408 on SOLVE-RATE not time. STEP 0 is mandatory and gating: (0a) a
  known-valid solved board MUST give E==0 or the energy is mis-specified (STOP,
  report per-constraint residual); (0b) the optimizer must solve EASY boards
  before any hard-board claim. Only then climb the optimizer ladder (annealing →
  random restarts → parallel tempering → block moves → Lagrangian). Reports
  solve_rate on ≥20 puzzles; timing only on the solved subset. A concrete
  instance of P0.1 on the cleanest "does energy-based global inference actually
  solve" testbed.

### Phase B — The architecturally-honest thesis test (depth)

- **exp3418 — Verifier-ensemble vs adaptive injection (HIGH, gemini).** Tests
  whether the full k=15 cross-mechanism ensemble beats the single distilled KAN
  (which collapsed to AUROC 0.475) on the SAME exp3273 held-out adaptive
  prompt-injection corpus. Reuses the built corpus + DeLong harness. DeLong vs
  the single KAN (does diversity help?) and vs the 20B teacher (replacement-grade?),
  with per-attack-category breakdown. Directly tests the Spera/Welch joint-null-space
  thesis on the domain where a lone verifier provably stalls.

### Phase C — Publication-gate closure (depth)

- **exp3419 — FoVer G2 reproduction harness (CRITICAL, claude).** Ships
  `scripts/reproduce_fover_headline.py` — a self-contained, fresh-clone,
  CPU-only reproducer per `ops/reproduction-runbook-fover-headline.md` — and runs
  it as a clean-room recompute (NOT reading existing `results/`) to confirm the
  headline lands in-CI (condition-A ∈ [0.9027, 0.9235], learning-contribution ∈
  [0.0125, 0.0245]). **Does NOT falsely claim G2 met** (that needs an external
  non-operator run); advances G2 from "documented runbook" to "turnkey +
  internally-confirmed," de-risking the external run.

### Phase D — Hardware (light, opportunistic per north-star §3)

- **exp3420 — KV260 terminal latency transcript (claude+opus).** Drive THE
  sovereignty board to its terminal state over SSH (NEVER host SD card), after
  which the per-milestone hardware mandate lifts for KV260.
- **exp3421 — GateMate bootstrap root-cause diagnostic (claude+opus).** NOT a
  fourth flash attempt (those recur "unspecified" = doomed rerun). A diagnostic
  of WHY the artifact never resolves a terminal verdict (toolchain vs board vs
  script). Opportunistic per north-star §3.
- **exp3422 — PolarFire reachability audit (claude, max_turns 20).** Light
  continuity check; no new workload. Opportunistic per north-star §3.

### Phase E — Ops (transition + depth-aligned synthesis)

- **exp3416 — archive .314 / activate .315.**
- **exp3423 — G1–G4 gate-status synthesis (gated on P0.1).** NOT a telemetry/
  evidence matrix vN+1 (the forbidden churn). Reports which gates moved + the
  P0.1 verdict + whether the forcing function can relax.
- **exp3424 — capstone v315 (gated on exp3423).** Honors the fabrication gate
  (skip flagged artifacts) and the Paper-v6 Narrowing Discipline.

---

## 6. Dependency graph

```
exp3416 (archive/activate)  ──► all depth tasks runnable

exp3312 (P0.1) ─────────────────────────────┐
exp3313 (P0.2) ─ independent                 │
exp3417 (Kona gate) ─ independent            ├─► exp3423 (gate status,
exp3418 (ensemble-vs-injection) ─ independent│      gated on exp3312.honest_verdict contains "complete")
exp3419 (G2 harness) ─ independent           │
                                             └─► exp3424 (capstone,
                                                    gated on exp3423.gate_status_v315_ready == true)

exp3420 / exp3421 / exp3422 (hardware) ─ independent, opportunistic
```

The depth tasks are mutually independent (each can run/fail without blocking the
others), so a single failure does not cascade. Only the ops synthesis +
capstone are gated, on cheap structured fields.

---

## 7. Hardware requirements

| Task | Substrate | Requirement |
|---|---|---|
| exp3312 P0.1 | `live_llm_inference` | CUDA (RTX 3090, recovered 2026-05-28); Qwen3.6-35B-A3B-GGUF AR baseline; Boltzmann-GPT substrate |
| exp3313 P0.2 | `verifier_ensemble_against_cached_candidates` | CPU + CUDA for model-based verifiers; FoVer corpus (committed) |
| exp3417 Kona | `live_llm_inference` | CPU Ising sampler + `adaptive_ising.py`; gemma-4-26B-A4B-GGUF only if hybrid LLM-proposal step used |
| exp3418 ensemble | `verifier_ensemble_against_cached_candidates` | CUDA for semantic/ThinkPRM verifiers; exp3273/exp3269 corpora (present) |
| exp3419 G2 | `verifier_ensemble_against_cached_candidates` | **CPU-only** (the point — no GPU/35B); FoVer corpus committed |
| exp3420 KV260 | `hardware_smoke` | `ssh kria` reachable; carnot_ising_v4 overlay |
| exp3421 GateMate | `hardware_smoke` | yosys/nextpnr-himbaechel/openFPGALoader diagnostic only |
| exp3422 PolarFire | `hardware_smoke` | `ssh polarfire` reachable |

SOTA local GGUF models used (per CLAUDE.md mandate): `unsloth/Qwen3.6-35B-A3B-GGUF`
(exp3312 AR baseline), `unsloth/gemma-4-26B-A4B-GGUF` (exp3417 hybrid step).

---

## 8. Discipline compliance

- **Depth-Over-Breadth Forcing Function (2026-05-30):** majority of substantive
  slots are P0.1/P0.2/Kona/ensemble/G2; zero `vN+1` re-measurement of an
  already-measured artifact. ✓
- **Failed-Experiment Rerun + Exclusion-Manifest Cross-Check:** every task that
  scope-matches a prior carries either a full 4-field `prior_failures:` block
  (exp3312, 3313, 3417, 3418, 3420, 3421) or an `operator_override:` citing the
  standing 2026-05-29 directive (routine transitions, hardware continuity,
  G2-runbook authorization). ✓
- **Verdict Terminal-Prefix:** every `honest_verdict` spec starts with
  `complete:`/`success:`/`passed:`/`shipped_`. ✓
- **Principle-Annotated Artifact Fields:** every REQUIRED ARTIFACT FIELD carries
  a `principle:`. ✓
- **Pre-Launch Preconditions:** every compute-bound task has a gating step-0
  precondition block with `blocked_*` fallbacks; KV260 uses SSH-not-SD-card. ✓
- **Inference-Substrate Declaration:** each task declares `inference_substrate`. ✓
- **Hardware-Task Continuity / north-star §3:** KV260 driven to terminal;
  GateMate/PolarFire opportunistic with operator_override citing north-star §3. ✓
- **Self-learning requirement (research-program.md):** exp3313 (α_t grounding)
  is the continuous-self-learning experiment. ✓
- **Operator-Only External Publication:** exp3419 explicitly does NOT flip
  `g2_independent_reproducer=true`; external run reserved for a non-operator. ✓

---

## 9. Success criteria for the milestone

The milestone **succeeds** (regardless of which way the science breaks) if:

1. **P0.1 reaches a recorded verdict** (premise validated / viable-not-superior /
   unsupported) — this alone relaxes the forcing function and is the highest-value
   outcome whether positive or negative.
2. **P0.2 reports λ_min(Σ) + effective-k** — telling us whether the grounding
   keystone holds at production k.
3. **The Kona gate reports a real solve_rate** (or an honest
   `blocked_energy_encoding_invalid` with the per-constraint residual).
4. **G2 reproduces in-CI** from a clean-room recompute, with the turnkey harness
   shipped.

It **fails** only if it reverts to breadth churn — re-measuring already-measured
artifacts instead of running the depth block.
