# Research Roadmap — Milestone 2026.05.322 (Depth-Over-Breadth VIII)

**Title:** P0.1 via Two Infra-Independent Routes — the Sudoku/Kona Correctness-First
Solve-Rate Gate (CPU, QUBO-validated) + the Energy-vs-Self-Consistency Crux on the In-Band
Contested Subset of Cached Corpora. Finally land a P0.1 verdict after three milestones lost
to infrastructure.

**Planner:** Claude Opus 4.8, 2026-05-31 (Depth-Over-Breadth Forcing Function ACTIVE).
**Milestone doc format:** follows v7/v8.
**Prior milestone:** 2026.05.321 (Depth-Over-Breadth VII).

---

## 1. What the previous milestone (.321) proved — and how it was lost

`.321` was **lost to infrastructure, not science.** Its design (a difficulty-matched
MATH levels-3/4 corpus → the energy-vs-SC crux in-band) was sound. The P0.1 crux failed
to run for the **THIRD consecutive milestone** (.319 GSM8K ceiling SC 0.908, .320 MATH-L5
floor SC 0.265, .321 infra) — but this time purely operationally:

| Exp | Result | Mechanism |
|---|---|---|
| exp3483 difficulty-matched corpus builder v2 | **FAIL → SKIP×2** | `API Error 400: 'thinking'/'redacted_thinking'` (opus extended-thinking 400 on a long live-GPU task) then pre-test self-heal failed. No corpus. |
| **exp3484 THE crux** | **SKIP×3 → RETIRED** | "Pre-tests failing, self-heal failed." **Energy-vs-SC never ran.** Its retirement cascade-GATE_BLOCKed exp3487/3491/3492. |
| exp3485 calibration v4 | **SKIP×3** | Pre-test self-heal failed. No artifact. |
| **exp3486 FR-11 minimal-β** | **CLEAN POSITIVE** | Found minimal-sufficient entropy β + grounding-dependence. The one landed depth result. |
| exp3487 Kona | **GATE_BLOCK** | Pre-emptive skip: upstream crux retired. |
| **exp3488 G2 package** | **CLEAN** | Regression-verified AUROC 0.9131, SHA256 + IPFS CID; external run pending. |
| exp3489 KV260 | `blocked_ssh_unreachable` | Board down (6 consecutive runs). |
| exp3490 PolarFire | **FLAGGED (tautology)** | Quarantined. |
| exp3491 synthesis / exp3492 capstone | **GATE_BLOCK** | Cascaded from the retired crux. |

**Three independent infra failure modes:** (a) the **`thinking`-400** on opus-routed
long/live tasks; (b) **pre-test self-heal failures** on heavyweight 80–100-turn tasks
whose scripts never reached a green state; (c) **idle-timeouts** ("Wall-clock+idle timeout
after 1201s (1201s silence)") on long silent optimization loops.

**What is now true:** G1/G3/G4 met; **G2 is the sole unmet publication gate** (operator-
gated external run). P0.1 **remains OPEN — not refuted.** Two large cached generation
corpora already exist on disk: `data/p01_gsm8k_generations.jsonl` (5.9 MB, SC ceiling
0.908) and `data/p01_hardmath_generations.jsonl` (3.4 MB, SC floor 0.265).

## 2. The .322 fix — architectural, not scientific

The science has been right for three milestones. The fix targets the loss mechanism:

1. **Route P0.1 through TWO infra-independent paths so no single fragile task gates the
   verdict.**
   - **Route 1 (PRIMARY, CPU-only, infra-bulletproof):** the Sudoku/Kona **correctness-
     first solve-rate gate** (the operator's explicitly-filed cleanest P0.1 testbed). No
     GGUF, no CUDA, no live generation → immune to the `thinking`-400 / CUDA / GGUF-
     tokenizer failure modes. Cross-validates Carnot's Ising-Sudoku encoding against the
     *verified-correct* published QUBO (arXiv:2403.04816) at Step 0a, then climbs the
     IRED/Kona adaptive-step optimizer ladder. Mandatory per-puzzle progress printing
     defeats idle-timeout.
   - **Route 2 (cached-scoring, infra-robust):** the energy-vs-SC crux on the **IN-BAND
     CONTESTED SUBSET** of the *already-cached* GSM8K/MATH corpora — the subset of problems
     whose per-problem SC lands in `[0.40, 0.70]`. This synthesizes headroom from existing
     data with **no fresh live build**, so the crux finally executes regardless of any live
     generation. Pure cached scoring → seconds → cannot idle-timeout, cannot `thinking`-400.
2. **Keep every task on sonnet (omit `model`).** The `thinking`-400 hit the **opus**-routed
   tasks (exp3483/exp3488); the sonnet tasks (exp3486/exp3490) ran clean. We deliberately
   deviate from "heavy → opus" because .321 proves opus `thinking`-400 is more dangerous
   than sonnet max-turns here, and the science is now routed through CPU/cached tasks that
   do not need opus-level reasoning. Tight scope + heavy harness reuse + generous max_turns
   mitigate max-turns (C+E escalation is skipped when `agent_type` is set).
3. **Break the cascade: no depth task is `gated_on` another depth task; the synthesis is
   UNGATED; only the capstone gates on the (robust, cached) synthesis.** The .321 cascade
   came from gating the synthesis/Kona on the fragile crux. Removing those gates means no
   depth-task retirement can pre-emptively GATE_BLOCK downstream.
4. **Print progress on every loop iteration** (corpus generation, optimizer restarts,
   self-learning iterations) to defeat the 1201 s idle-timeout.
5. **gemini-cli is DOWN** (.315/.316/.318 crashed; .321 planning crashed) → ALL tasks on
   `agent_type: claude`, `requires_claude: true`.

## 3. The three biggest PRD-vision gaps this milestone targets

1. **The Kona/Phase-3 premise (P0.1) has never produced a verdict.** The entire
   foundation-model endgame assumes energy-descent global inference beats/matches
   autoregressive token sampling. Three milestones tried and lost to infra. .322 lands a
   verdict via two robust routes.
2. **Continuous self-learning (FR-11) needs a deployable grounding law.** .321 found a
   minimal-sufficient entropy β; the open question is whether β_min is *predictable* from
   the measured verifier-ensemble diversity (λ_min(Σ), the P0.2 keystone). exp3498 fuses
   the FR-11 minimal-β finding with the λ_min audit into a predictive Phase-5 default —
   the mandatory continuous-self-learning task, doubling as the P0.2 follow-up.
3. **Publication convergence (G2).** G2 is the sole unmet gate. exp3499 keeps the
   regression-clean package bulletproof and the external ask one-click, while never
   pushing/triggering (Operator-Only External Publication).

## 4. Architecture / dependency graph (cascade-proof)

```
exp3493 archive .321 / activate .322  (ops)
        |
        v
   +--------------- DEPTH BLOCK (no cross-gating) ---------------+
   |  exp3494  P0.1 Route 1: Sudoku/Kona correctness-first       |  CPU, no LLM   (PRIMARY)
   |           solve-rate gate (QUBO-validated, IRED ladder)     |
   |  exp3495  P0.1 Route 2: energy-vs-SC crux on the IN-BAND    |  cached scoring
   |           CONTESTED SUBSET of cached GSM8K/MATH corpora     |
   |  exp3496  (OPTIONAL, non-blocking) live difficulty-matched  |  live GGUF, sonnet
   |           corpus builder v3 (resume exp3483 script)         |
   |  exp3497  MATH-aware energy-correctness calibration v5      |  cached scoring
   |           (tautology-FIXED, step-vs-final AUROC)            |
   |  exp3498  FR-11 beta_min = f(lambda_min) predictive law     |  cached  (self-learning + P0.2)
   +------------------------------------------------------------+
   exp3499  G2 clean-room regression-verify + external-ask refresh  (cached; never pushes)
   exp3500  KV260 terminal latency transcript (SSH precondition)    (hardware, opportunistic)
   exp3501  PolarFire opportunistic reachability audit              (hardware, opportunistic)
        |
        v
   exp3502  G1-G4 gate-status synthesis v322   (UNGATED -- reads & skips absent/flagged)
        |  (gated_on exp3502.gate_status_v322_ready == true)
        v
   exp3503  Capstone v322
```

Every gate in the chain depends only on the robust cached synthesis, never on a P0.1
task that might retire on infra. exp3496 is non-blocking; its failure cannot cascade.

## 5. Phases

- **Phase A — OPS:** exp3493 archive/activate.
- **Phase B — DEPTH (majority of slots, Depth-Over-Breadth):** exp3494 (Sudoku P0.1,
  primary), exp3495 (crux, contested subset), exp3496 (optional live builder),
  exp3497 (calibration), exp3498 (self-learning / P0.2 law).
- **Phase C — G2 (sole publication gate):** exp3499.
- **Phase D — HARDWARE (opportunistic continuity, north-star §3):** exp3500 KV260,
  exp3501 PolarFire.
- **Phase E — SYNTHESIS:** exp3502 gate status, exp3503 capstone.

## 6. Hardware requirements

- exp3494/3495/3497/3498/3499/3502/3503: **CPU only** (no GPU). This is the point — the
  P0.1 verdict no longer depends on the GPU/live-LLM stack.
- exp3496 (optional, non-blocking): 1× RTX 3090 for live GGUF generation; gated behind
  CUDA + GGUF-tokenizer preconditions; blocks cleanly if unavailable.
- exp3500/3501: SSH reachability to KV260 (`kria`) / PolarFire; honest blocked verdict if
  unreachable.

## 7. Depth-Over-Breadth compliance

Per the 2026-05-30 forcing function, the milestone does NOT relax (P0.1 has no clean
verdict; G2 not externally run). Every task advances the headline, closes a G-gate, or
tests a load-bearing-unproven link. **No vN+1 re-measurement of an already-measured
artifact:** exp3494 is the *correctness-first, QUBO-validated, ladder-climbing* Sudoku gate
(new vs exp3408's time-framed plateau and exp3475's saturated block); exp3495 runs the
energy-vs-SC comparison on a *contested-subset* corpus that finally satisfies the headroom
precondition (the question exp3472/exp3484 structurally could not answer); exp3498 derives
a *predictive law* beta_min=f(lambda_min) (new vs exp3486's single-point minimal-β). The
forcing function relaxes only once P0.1 has a clean verdict AND G2 is externally in-motion.

## 8. SOTA model usage

The only live-LLM task (exp3496, optional) uses `unsloth/gemma-4-26B-A4B-it-GGUF`
(default) with `unsloth/gemma-4-31B-it-GGUF` documented fallback, via the GGUF model_path
loader (embedded tokenizer; NEVER `AutoTokenizer` on a `-GGUF` repo id, per the 2026-05-29
GGUF tokenizer rule). All cached-scoring tasks consume the already-generated corpora.

## 9. Cross-references

- `ops/north-star.md` — headline (§1), G1–G4 gate (§2), KV260-terminal hardware focus (§3)
- `ops/known-issues.md` — KONA GLOBAL-OPT CORRECTNESS-FIRST GATE (2026-05-30), PHASE-3
  PATH DE-RISKING ROADMAP / exp3312 P0.1 + exp3313 P0.2 (2026-05-29)
- CLAUDE.md — Depth-Over-Breadth Forcing Function, Failed-Experiment Rerun Discipline,
  Pre-Launch Preconditions, Adversarial Artifact Verification (fabrication gate)
- research-references.md "2026-05-31 Post-.321 Planning Sweep" (QUBO-Sudoku 2403.04816,
  IRED 2406.11179, Enso, Closed-Loop Transformers 2511.21882, Kona datapoint)
- exp2837 (FoVer 0.9131 headline), exp3486 (FR-11 minimal-β), exp3439 (λ_min audit v3)
