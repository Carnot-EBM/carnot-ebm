# Research Roadmap — Milestone 2026.05.316

**Depth-Over-Breadth II: Clean the Existential Block + Reframe P0.1 Around Self-Consistency**

Planner: Opus 4.8, 2026-05-30. Supersedes the `.315` design doc in place.

---

## TL;DR

`.315` was the FIRST Depth-Over-Breadth milestone and it pointed the loop at the
right targets — but **3 of its 5 load-bearing depth tasks did not cleanly land**.
`.316` is not a new direction; it is the **honest completion of the same
existential block**, plus a sharpened P0.1 that answers the question `.315`
actually surfaced.

The Depth-Over-Breadth Forcing Function (CLAUDE.md, 2026-05-30) does **NOT**
relax for `.316`. The `.315` capstone reported `depth_forcing_function_can_relax
= true`, but that was premature: P0.1's verdict is **contested and
flagged_adversarial**, P0.2 never ran, the Kona gate never ran, and the
ensemble-injection test never ran. The function relaxes only once P0.1 has a
**clean** verdict AND G2 has a concrete in-flight reproducer. `.316` is built to
earn that.

---

## What `.315` proved (and didn't)

| Depth task | Outcome | Status going into `.316` |
|---|---|---|
| **P0.1** energy-descent vs AR (exp3312) | Landed, **flagged_adversarial**. Energy-selection 0.840 beat *greedy* AR 0.750 (+0.090, McNemar p=0.033) but **LOST to equal-compute self-consistency** (majority vote, 0.895, Δ −0.055). The "energy-descent" was best-of-3 reranking + 8 latent steps. | **Contested.** Rerun clean + reframe around the SC control. |
| **P0.2** verifier-diversity / λ_min(Σ) (exp3313) | **Displaced** — deliverable became a repair-substrate autopsy; the diversity audit never ran. | **Never ran.** Re-attempt. |
| **Kona** solve-rate gate (exp3417) | **No artifact produced.** | **Never ran.** Re-attempt. |
| **Ensemble-vs-injection** (exp3418) | **No artifact produced.** | **Never ran.** Re-attempt. |
| **G2** reproduction harness (exp3419) | **Clean landing.** `scripts/reproduce_fover_headline.py` reproduces condition-A AUROC 0.9131 + learning-contribution 0.0185 in-CI from a clean recompute. | Advance to **isolated clean-room** validation; external run still operator-gated. |

The single most important takeaway: **at equal compute, energy-based selection
did not beat plain majority-vote self-consistency** on GSM8K. That is the real
test of the Kona premise, and `.316` is built around it.

---

## The three biggest gaps between current state and the PRD vision

1. **The Kona premise is unproven at equal compute.** Phase-3's foundation-model
   endgame assumes energy-descent reasoning on continuous latents *beats* token
   sampling. `.315` showed it beats *greedy* AR but loses to *self-consistency* —
   and the result was flagged. We do not yet know if energy adds anything beyond
   what majority vote already gives you. **Gap closed by exp3426 (P0.1 v2).**

2. **The α_t grounding keystone is unmeasured at production k.** The entire
   self-correcting-model thesis (PRD FR-11 / continuous self-learning) rests on
   the verifier ensemble having real diversity (small joint null space). exp1224
   showed k=3 collapsing to effective k=1, and the FoVer headline showed 3 of 4
   verifiers contributing zero. We have never measured λ_min(Σ) on a deliberately
   disjoint-kernel suite at larger k. **Gap closed by exp3427 (P0.2).**

3. **Energy-based global inference has never been shown to *solve*, only to run
   fast.** exp3408's Sudoku run descended energy 2104→10 but `solved=False`,
   making its "15× speedup" fast-but-wrong. We do not know whether Carnot's Ising
   energy formulation can actually solve a hard combinatorial reasoning task.
   **Gap closed by exp3428 (Kona solve-rate gate).**

---

## Architecture under test

```
          ┌─────────────────────── P0.1 (exp3426): the premise ────────────────────────┐
          │  problem ──► base LLM (Qwen3.6-35B-A3B-GGUF)                                  │
          │                 │                                                            │
          │     ┌───────────┼───────────────┬────────────────────┐                      │
          │     ▼           ▼                ▼                    ▼                      │
          │  greedy AR   k samples ──►   k samples ──►       k samples ──►               │
          │  (control)   majority vote   self-certainty       ENERGY-weighted vote       │
          │              (SC, the real    BoN (2502.18581)     + latent energy descent   │
          │               control .315    strongest cheap       (EBM-CoT 2511.07124)     │
          │               lost to)        selector)             ── the premise           │
          │     └───────────┴───────────────┴────────────────────┘                      │
          │  headline = energy-vote accuracy − SC accuracy (paired, matched compute)     │
          └──────────────────────────────────────────────────────────────────────────────┘

   P0.2 (exp3427): does the ensemble that GROUNDS self-correction have real diversity?
       173 verifiers ──► label by kernel class ──► Σ (decision covariance)
       ──► λ_min(Σ), effective-k (participation ratio), drop-one-out contribution
       grounding holds ⇔ λ_min > 0.1 AND effective-k ≥ 3

   exp3428 (Kona, concrete P0.1 instance): can energy SOLVE, not just run fast?
       STEP 0a encoding validity (valid board ⇒ E==0)  ──gate──►  optimizer ladder
       ──► solve_rate by difficulty (final_energy==0 verified ON THE BOARD)

   exp3429: does the k=15 cross-mechanism ensemble beat a lone KAN (0.475) on the
            adaptive prompt-injection corpus where any single mechanism fails?

   G2 (exp3430): run the shipped harness in an ISOLATED clean-room env (fresh
                 worktree + fresh venv) to de-risk the external reproducer.
```

---

## Phases

### Phase A — OPS transition (1 task)
- **exp3425** — archive `.315` honestly (note the contested/flagged P0.1 + the
  3 non-landing depth tasks), activate `.316`.

### Phase B — DEPTH BLOCK (the majority; 5 tasks)
The load-bearing existential tests, completed cleanly.
- **exp3426 — P0.1 v2 (THE crux):** energy-descent vs AR vs **self-consistency**
  vs self-certainty-BoN, at matched compute, clean methodology to clear the
  adversarial flag. Headline = energy-weighted-vote − majority-vote SC. Mirrors
  EBM-CoT (2511.07124). `requires_claude`.
- **exp3427 — P0.2:** verifier-ensemble λ_min(Σ) / joint-null-space diversity
  audit on a deliberately disjoint-kernel suite. **This is the milestone's
  continuous-self-learning depth experiment** (α_t grounding precondition for
  FR-11 self-correction). `gemini`.
- **exp3428 — Kona solve-rate gate:** correctness-first (STEP 0a encoding
  validity gates everything), solve_rate not time. `gemini`.
- **exp3429 — ensemble-vs-adaptive-injection:** full k=15 cross-mechanism
  ensemble on the exp3273 held-out corpus the lone KAN scored 0.475 on. `gemini`.
- **exp3430 — G2 clean-room validation:** run `scripts/reproduce_fover_headline.py`
  in an isolated fresh git worktree + fresh venv; confirm in-CI; document the
  turnkey external path. Does NOT set `g2_independent_reproducer=true` (external
  non-operator run still required). `claude`.

### Phase C — HARDWARE (light + opportunistic per north-star §3; 3 tasks)
KV260 to terminal then freeze; GateMate/PolarFire opportunistic, never blocking.
- **exp3431** — KV260 terminal latency transcript (re-attempt; was
  `blocked_kv260_ssh_unreachable` in `.315`). SSH-only precondition.
- **exp3432** — GateMate: apply the `.315`-identified fix (exp3421 root-cause =
  script never sets `honest_verdict`) to the experiment script, then attempt the
  bootstrap once. Opportunistic.
- **exp3433** — PolarFire light reachability/continuity audit (no new workload).

### Phase D — OPS synthesis + capstone (2 tasks)
- **exp3434** — G1–G4 gate-status synthesis (gated on exp3426 P0.1 v2 verdict).
- **exp3435** — Capstone v316 (gated on exp3434).

---

## Dependency graph

```
exp3425 (archive/activate)
   │
   ├─► exp3426  P0.1 v2  ──────────────┐  (gates exp3434 on honest_verdict contains 'complete')
   ├─► exp3427  P0.2 diversity         │
   ├─► exp3428  Kona solve-rate        │
   ├─► exp3429  ensemble-vs-injection  │
   ├─► exp3430  G2 clean-room          │
   ├─► exp3431  KV260 terminal         │
   ├─► exp3432  GateMate fix-apply     │
   └─► exp3433  PolarFire audit        │
                                       ▼
                              exp3434  G1–G4 synthesis  (gated_on exp3426 verdict)
                                       │  (gates exp3435 on gate_status_v316_ready==true)
                                       ▼
                              exp3435  Capstone v316
```

---

## Hardware requirements

- **2× RTX 3090 (CUDA):** exp3426 (live 35B GGUF inference + latent energy
  descent), exp3427 / exp3429 (model-based verifiers), exp3428 (Ising sampler +
  optional LLM proposal step). CUDA recovered 2026-05-28 — no compute blocker.
- **CPU only:** exp3430 (FoVer verifier-scoring reproduction), all OPS tasks.
- **KV260** via `ssh kria` (SSH-only precondition, never host SD-card).
- **GateMate** via `openFPGALoader -c dirtyJtag`; **PolarFire** via `ssh polarfire`.

## SOTA model usage (CLAUDE.md mandate)

- exp3426 AR + energy-descent baseline: `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship
  MoE), loaded via the GGUF path (embedded tokenizer — NEVER `AutoTokenizer` on a
  `-GGUF` repo id, per the 2026-05-29 GGUF tokenizer rule).
- exp3428 optional hybrid LLM-proposal step: `unsloth/gemma-4-26B-A4B-it-GGUF`.
- exp3427 / exp3429 model-based verifiers use the cached SOTA pair as configured
  in the verifier suite.

## Self-learning coverage (PRD / research-program.md mandate)

**exp3427 (P0.2) is the continuous-self-learning experiment.** α_t grounding is
the precondition that lets a self-correcting model (FR-11, Tiers 1–4) avoid
self-distillation collapse; it only holds if the verifier ensemble has real
diversity. exp3427 measures exactly that keystone at production k. The FR-11
learning-contribution (+0.0185, exp2837) is the live self-learning signal whose
grounding exp3427 validates.

## Depth-Over-Breadth compliance statement

Every `.316` substantive task either (a) tests a load-bearing-unproven link
(P0.1 exp3426, P0.2 exp3427, Kona exp3428), (b) closes/advances a publication
gate (G2 exp3430, gate synthesis exp3434), or (c) tests whether the ensemble
generalizes where a single verifier failed (exp3429). **No task re-measures an
already-measured artifact** (no cross-corpus matrix vN+1, no telemetry vN+1, no
repair-panel vN+1). P0.1 v2 is NOT re-measurement — the `.315` result was flagged
and contested, and the v2 adds the matched-compute self-consistency control as
the PRIMARY comparison (a question the `.315` run did not answer). The hardware
tasks are the light, opportunistic, KV260-to-terminal allocation north-star §3
prescribes.
