# Research Roadmap — Milestone 2026.05.324

**Depth-Over-Breadth X: Harden the P0.1 Sudoku Positive into a Defensible Result
(fair LLM-AR baseline + PT diagnosis) + Generalize It to a SECOND CSP +
Fix the Route-2 Energy-Reranker Collapse + De-Flag the Step-to-Final Gap +
Find the Self-Learning Rule That Actually Deploys + G2 Drift-Verify**

**Planner:** Claude Opus 4.8 — 2026-05-31
**Milestone doc for:** `research-roadmap-next.yaml` (milestone `2026.05.324`)
**Prior milestone:** `2026.05.323` (Depth-Over-Breadth IX)

---

## 1. What the previous milestone (.323) proved

`.323` was the milestone P0.1 (does energy-based *global* inference actually
**solve**, not just descend fast?) produced its **first clean positive datapoint** —
and simultaneously exposed which parts of that positive are fragile and which
adjacent results are broken.

| Experiment | Verdict | Status |
|---|---|---|
| **exp3505** P0.1 Route 1 — Sudoku real-optimizer ladder | `solve_rate=1.0` (discrete SA / 20-restarts / exact-CP all 1.0), AR greedy `0.0`, vanilla Langevin `0.0`, encoding E==0 re-asserted | **CLEAN POSITIVE** |
| **exp3507** P0.1 Route 2 — energy-vs-SC on in-band level-3 corpus | every energy metric == SC baseline `0.653061`, `flip_count=0`, `delta=0.0` | **FLAGGED — real reranker collapse** |
| **exp3508** step-to-final gap closure | `min` aggregation 0.601→0.903 (`gap_closed=0.9665`) BUT stored reference==measured twice | **FLAGGED — directional only** |
| **exp3509** FR-11 β-law deployment | `deployed_law_prevents_collapse=False` | **CLEAN NEGATIVE** |
| **exp3510** G2 regression-verify | package reproduces 0.9131 within CI; `g2_met=False` | **CLEAN; external pending** |
| **exp3506** level-3 corpus extend | `n=49` (in band), partial | scorable, < headline |
| exp3513 / exp3514 synthesis + capstone | `g1∧g3∧g4` met, `unmet_gates=['G2']`, `depth_forcing_function_can_relax=True` | clean (seed-fix worked) |

**The load-bearing finding:** with the encoding *validated* (E==0) and a *proper
combinatorial optimizer* (discrete SA / restarts / exact-CP), energy-based global
inference solves Sudoku at 100% where autoregressive generation solves 0%. This is
the first clean piece of P0.1 evidence — but it is **fragile** (21 puzzles, a naive
greedy AR baseline, PT inexplicably at 0.38) and **narrow** (Sudoku only), and its
more product-relevant sibling (energy-vs-self-consistency on natural-language math)
is **broken** (the reranker collapsed onto the SC majority).

**Why .324 stays in DEPTH.** Although `depth_forcing_function_can_relax=True`, the
P0.1 positive is not yet defensible and Route 2 does not work. Relaxing into breadth
churn now would squander the first positive. Per the Depth-Over-Breadth Forcing
Function, every .324 task answers a question its predecessor structurally could not —
no `vN+1` re-measurement.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The P0.1 positive is not yet a defensible result.** A 21-puzzle Sudoku win
   against a *naive greedy* AR baseline, with parallel tempering mysteriously
   underperforming simulated annealing, is a promising datapoint — not a
   headline-class claim. Gap: a fair LLM-AR baseline (the literature has it:
   Sudoku-Bench <15%, Kona 96% vs LLMs 2%, Pathway BDH 97.4% vs ~0%), more puzzles,
   harder tiers, and a PT diagnosis — **and a SECOND CSP** to show this is a general
   property of energy-global-inference-vs-AR, not Sudoku-overfit (the discrete-
   diffusion-beats-AR literature, arXiv:2410.14157, did exactly this across Sudoku /
   SAT / Countdown).

2. **Route 2 — energy-vs-SC on real natural-language reasoning — is broken.** This is
   the product-relevant test (does the verifier energy improve answer *selection*
   over self-consistency?). exp3507's reranker collapsed onto the SC majority
   (`flip_count=0`). This is the "consensus trap" (CoVerRL, arXiv:2603.17775). Gap:
   a reranker that produces *distinct* selections and is measured tautology-clean,
   then honestly compared to SC and the ThinkPRM bar.

3. **Continuous self-learning has no deployable rule yet.** The offline-fitted
   `β_min=f(λ_min)` law does NOT generalize to deployment (exp3509). Gap: validate
   the conservative-default β's robustness across many fresh configs AND test an
   ADAPTIVE ONLINE β rule (measure λ_min online, clamp to a safe floor) as the
   deployable Phase-5 self-learning default.

(The fixed finish line — G2, an independent reproducer — is the sole unmet
publication gate and is kept fresh by a drift-verify task; it is operator-gated.)

---

## 3. Architecture of the milestone

```
                         .324 — Depth-Over-Breadth X
                         (P0.1 positive: HARDEN + GENERALIZE; Route 2: FIX)

  PHASE A (ops)         exp3515  archive .323 / activate .324  [seed 20260531]

  PHASE B (depth — no cross-gating; all CPU/cached except the optional live builder)
    Corpus (opt) ──  exp3516  extend level-3 in-band corpus to n>=80  [live GGUF, NON-BLOCKING]
    P0.1 Route 1 ─┬─ exp3517  HARDEN Sudoku positive (CPU core + CUDA-optional LLM-AR)
                  └─ exp3518  GENERALIZE to a 2nd CSP (graph-coloring / SAT / Countdown)
    P0.1 Route 2 ──  exp3519  FIX the energy-reranker collapse (break the consensus trap)
    Mechanism    ──  exp3520  DE-FLAG + verify the step-to-final gap (distinct fields + shuffle control)
    Self-learning──  exp3521  conservative-default + ADAPTIVE-ONLINE β  [MANDATORY self-learning + P0.2]

  PHASE C (gate)       exp3522  G2 drift-verify (no push, operator-gated)

  PHASE D (hardware)   exp3523  KV260 terminal latency transcript (SSH precondition)
                       exp3524  PolarFire opportunistic reachability audit

  PHASE E (synthesis)  exp3525  G1–G4 gate-status synthesis v324  [UNGATED, cascade-proof, seed 20260531]
                       exp3526  Capstone v324  [gated ONLY on exp3525 ready, seed 20260531]
```

### The five rules carried from the working .323 architecture (+ the tightened anti-tautology rule)

1. **ALL tasks `agent_type: claude`, `requires_claude: true`** — gemini-cli is DOWN
   (known-issues 2026-05-01 + the .322/.323 operational reality); claude is the only
   backend landing artifacts.
2. **NO `model: opus` anywhere** — the opus extended-thinking `thinking`-400 killed
   .321's builder and .322's first G2. Everything is sonnet.
3. **CASCADE-PROOF** — no depth task is `gated_on` another depth task; exp3519 does
   NOT gate on exp3516 (it reads whatever level-3 corpus exists, >=49 cached); the
   synthesis (exp3525) is UNGATED and reads-and-skips absent/flagged artifacts; only
   the capstone gates on the synthesis-ready flag.
4. **PER-ITERATION progress flush + a hard wall-clock budget** on every loop — defeats
   the 1201s idle-timeout that hit .321/.322 silent loops.
5. **Anti-tautology seed + distinct-field discipline** (the .323 lesson, tightened):
   - aggregation/synthesis/capstone tasks set `random_seed = 20260531` (a distinct
     fixed value, NEVER the experiment number — the exp3502/3503 flag);
   - measurement tasks set a CONTENT-DERIVED `random_seed`, never the exp number;
   - tasks MUST NOT store a *reference* value in a field that is bit-identical to a
     *measured* field (the exp3508 flag) — references go in a `methodology_note`
     string, not a duplicate numeric field;
   - tasks MUST NOT let every energy/selection metric collapse to the SC baseline
     (the exp3507 flag) — a runtime assert requires the energy condition arrays to be
     element-wise distinct from the SC baseline array (or the verdict is the honest
     `blocked_reranker_degenerate`).

---

## 4. Phase descriptions

### Phase A — OPS transition (exp3515)
Archive .323 honestly to `ops/changelog.md`, write the .323 operational retrospective
JSON, confirm .324 active. `seed=20260531`.

### Phase B — DEPTH BLOCK (exp3516–exp3521)

- **exp3516 — Extend level-3 in-band corpus to n>=80 (live GGUF; OPTIONAL, NON-BLOCKING).**
  Resume the exp3506 builder; per-problem flush + hard 18-min budget. Nothing gates on
  it (exp3519 runs on whatever exists, >=49 cached). Satisfies the live-SOTA-GGUF
  generation slot. If CUDA is down, blocks honestly with no cascade.

- **exp3517 — P0.1 Route 1 HARDENING (CPU core; CUDA-optional LLM-AR).** Re-run the
  validated-encoding optimizer ladder on **>=40 puzzles** across harder tiers
  (Sudoku-Bench difficulty metric). DIAGNOSE why parallel tempering underperformed
  SA in exp3505 (temperature ladder / swap-acceptance instrumentation). Report a
  **fair AR baseline**: the documented literature numbers (Sudoku-Bench <15%,
  Kona 96% vs LLM 2%, BDH 97.4% vs ~0%) AND an optional in-house SOTA-GGUF LLM-AR run
  on the same puzzles (CUDA-gated, non-fatal — the CPU core lands regardless).
  Time-to-solution vs AR only on the solved subset. THE primary hardening task.

- **exp3518 — P0.1 Route 1 GENERALIZATION to a SECOND CSP (CPU).** Apply the SAME
  energy-global-inference-vs-AR + exact-baseline + solve-rate protocol to a *different*
  combinatorial reasoning family (graph coloring and/or Boolean SAT and/or Countdown,
  per arXiv:2410.14157 + DIFUSCO). An exact solver confirms instances are solvable;
  solve-rate is the gate; AR is the comparator. Shows the Sudoku positive is a general
  property, not Sudoku-overfit.

- **exp3519 — P0.1 Route 2 SUBSTRATE FIX (cached).** Fix the energy-reranker collapse
  (every metric == SC baseline, λs collapsed to 0). Diagnose the degeneracy, break the
  consensus trap (CoVerRL), and re-run the energy-vs-SC crux on the level-3 in-band
  corpus (>=49 cached). Tautology-clean by construction: a runtime assert requires the
  reranker's selection array to differ from the SC majority on >=1 problem
  (`flip_count>0`) or the verdict is the honest `blocked_reranker_degenerate`. Clears
  (or honestly fails to clear) the ThinkPRM bar.

- **exp3520 — DE-FLAG + verify the step-to-final gap (cached).** Re-run the
  aggregation sweep (last/product/min/mean/uncertainty-weighted) with DISTINCT field
  names (no reference==measured duplication) AND a **shuffle-label negative control**:
  shuffle the per-step labels and confirm the `min`-aggregation AUROC collapses — if
  it does NOT collapse, the 0.97 gap-closure was a label-correlated tautology, not a
  mechanism. Confirms whether exp3508's 97% gap closure is real.

- **exp3521 — Robust self-learning default (cached) [MANDATORY continuous-self-learning
  + P0.2].** Since the offline-fitted β-law failed to deploy (exp3509), validate the
  conservative-default β's robustness across many fresh grounding configs AND test an
  ADAPTIVE ONLINE β rule (measure λ_min online during the FR-11 loop, set
  β=clamp(f(λ_min), β_floor)) — does an online rule prevent depth-N>=200 collapse where
  the static offline law did not? Arms: A (adaptive-online), B (β=0, expect collapse),
  C (conservative-default). pass_rate and true_accuracy kept distinct (runtime assert).

### Phase C — G2 (exp3522)
Clean-room regression-verify the self-contained FoVer package still reproduces 0.9131
within CI after .324's changes (drift detection); keep the one-click external-ask
current. NEVER pushes / triggers CI / flips G2 (Operator-Only External Publication).

### Phase D — HARDWARE (exp3523–exp3524)
KV260 terminal latency transcript via the SSH precondition (north-star §3: drive to
terminal, then freeze; SSH-unreachable 9+ runs — honest blocked verdict if still
down). PolarFire opportunistic reachability audit with strictly-distinct fields.
GateMate is opportunistic per north-star §3 and is not blocked-on this milestone.

### Phase E — SYNTHESIS (exp3525–exp3526)
G1–G4 gate-status synthesis (UNGATED, cascade-proof, reads-and-skips absent/flagged,
`seed=20260531`) and capstone (gated only on the synthesis-ready flag, `seed=20260531`).

---

## 5. Dependency graph

```
exp3515 (archive)
   │ (no hard dependency; ops gate)
   ▼
exp3516 ── exp3517 ── exp3518 ── exp3519 ── exp3520 ── exp3521 ── exp3522 ── exp3523 ── exp3524
 (all independent; NO depth task gated_on another depth task — cascade-proof)
   │
   ▼
exp3525 (gate synthesis — UNGATED: reads whatever landed, skips absent/flagged)
   │  gated_on: exp3525.gate_status_v324_ready == true
   ▼
exp3526 (capstone)
```

The only structured gate in the milestone is `exp3526 gated_on exp3525.gate_status_v324_ready`.
exp3519/exp3520 handle a small/absent corpus via their own PRECONDITIONS. This is the
exact cascade-proof shape that let .322/.323 land after .321's gate-cascade loss.

---

## 6. Hardware requirements

| Task | Substrate | GPU? | Notes |
|---|---|---|---|
| exp3516 | live GGUF generation | yes (CUDA-gated) | OPTIONAL/non-blocking; blocks honestly if CUDA down |
| exp3517 | CPU global-opt core; CUDA-optional LLM-AR | optional | CPU core lands regardless; GGUF AR baseline CUDA-gated, non-fatal |
| exp3518 | CPU global-opt (2nd CSP); CUDA-optional LLM-AR | optional | same pattern |
| exp3519/3520/3522 | cached verifier scoring / cached traces | no | seconds; cannot thinking-400 / idle-timeout |
| exp3521 | cached-trace closed-loop sweep | no | seconds |
| exp3523 | KV260 over SSH | no | `ssh kria` reachability precondition (NOT host SD card) |
| exp3524 | PolarFire over SSH | no | `ssh polarfire` reachability precondition |
| exp3515/3525/3526 | aggregation | no | sub-second |

SOTA local GGUF models for the LLM-using tasks (exp3516/3517/3518 baselines):
`unsloth/gemma-4-26B-A4B-it-GGUF` (default) / `unsloth/gemma-4-31B-it-GGUF` (fallback),
loaded via the GGUF path (embedded tokenizer; NEVER `AutoTokenizer` on a `-GGUF` repo id).

---

## 7. Compliance checklist

- **Depth-Over-Breadth:** every task answers a NEW question (harden / generalize / fix /
  de-flag / find-deployable-rule); no `vN+1` re-measurement of an already-measured artifact.
- **Continuous self-learning:** exp3521 (mandatory) — adaptive-online β after the offline law failed.
- **SOTA GGUF models:** exp3516/3517/3518 include the mandated GGUFs in MODEL_SPECS (CUDA-gated).
- **Hardware-Task Continuity / north-star §3:** KV260 (until terminal) + PolarFire opportunistic.
- **Verdict Terminal-Prefix:** every `honest_verdict` starts `complete:`.
- **Pre-Launch Preconditions:** every compute-bound task has a PRECONDITIONS step 0.
- **Anti-fabrication:** content-derived seeds on measurements, `20260531` on aggregations;
  no reference==measured duplicate fields; distinct-from-SC runtime asserts on rerankers.
- **Operator-Only External Publication:** exp3522 never pushes / triggers CI / flips G2.
- **Public Documentation Discipline:** no task edits `docs/index.html`, README, or roadmap prose.
