# Research Roadmap — Milestone 2026.05.320

**Title:** Depth-Over-Breadth VI — P0.1 on a Benchmark With Headroom: Does Energy
Reranking Beat Self-Consistency Where SC Isn't at Ceiling? + G2 Self-Contained
External Package

**Planner:** Claude Opus 4.8 (2026-05-30), under the Depth-Over-Breadth Forcing
Function (MANDATORY, 2026-05-30) and the north-star (`ops/north-star.md`).

---

## 1. What the previous milestone (.319) proved

`.319` (Depth-Over-Breadth V) ran the decisive **trained-energy** P0.1 test. The
answer is now sharp, clean where it matters, and it points directly at this
milestone's design.

| Experiment | Result |
|---|---|
| exp3459 corpus | GSM8K P0.1 corpus reached the full **n=120** (resumable; SC non-degenerate 0.90). Headline-eligible. |
| exp3461 calibration v2 (CLEAN) | **Trained** EORM energy reaches `correctness AUROC = 0.629` — a **+0.113 lift** over the 0.516 untrained floor. FoVer step-error energy 0.606. **Training fixes the uninformative-energy problem.** |
| exp3460 P0.1 v5 (FLAGGED) | n=120 held-out, k=6, 5-fold CV: greedy 0.850, **SC 0.9083**, self-certainty-BoN 0.842, FoVer-energy-argmin 0.825 (**LOSES**, −0.083, p=0.002), trained-energy-weighted-vote **0.9083 = SC EXACTLY** (delta 0.0, p=1.0), hybrid 0.9083 = SC. **Verdict: a trained energy MATCHES but does NOT BEAT SC at matched compute.** |
| exp3464 Kona | Trained energy **no lift** over the untrained hybrid (both solve 1.0 — benchmark saturated). |
| exp3463 G2 | CI workflow dry-run green + Docker clean-room reproduces AUROC 0.9131 + handoff package ready. **External/non-operator run still pending — G2 remains the SOLE unmet gate.** |
| exp3462 FR-11 | At N=50 self-improvement iterations, ARM A did **not** mode-collapse (residual diversity holds); finding informative but FLAGGED again for a tautology. |
| Hardware | KV260 `blocked_kv260_ssh_unreachable`; GateMate toolchain missing; PolarFire reachable. |

**Gate status entering .320:** G1 ✅, G3 ✅, G4 ✅, **G2 ❌ (sole unmet)**.
Depth-Over-Breadth **cannot relax** — P0.1's verdict is directionally clear but
`flagged_adversarial` (not clean), and G2's external run is still pending.

## 2. The diagnosis that drives .320

**GSM8K self-consistency is at CEILING (0.908).** A near-perfect majority vote leaves
almost no room for any selector to help: the energy-weighted vote degenerates onto the
majority answer (hence the *exact* tie, McNemar p=1.0). The recurring TAUTOLOGY flag is
the **symptom** of testing the selection premise on a saturated benchmark, not a code
bug — exp3461 proves the energy itself now carries real signal (AUROC 0.629).

The literature confirms the mechanism and the fix:

- **arXiv:2602.11570 (PRIME):** process-aware verification beats outcome-only by
  **+8–9% on AIME** — the gains appear on HARD benchmarks with headroom, from
  step-level verification.
- **Sweep nuance:** majority voting can outperform PRM-Best-of-N because *verifiers
  fail to identify the minority-yet-correct solution*. On a ceiling benchmark the
  minority-correct fraction is tiny → nothing to recover → energy ties SC. On a
  headroom benchmark it is large → a good process verifier has room to win.
- **arXiv:2510.13918:** optimal SC+PRM aggregation can exceed either signal alone.

## 3. The .320 hypothesis (a genuinely new question, not vN+1 churn)

> On a benchmark **with headroom** (hard math where SC ≈ 0.4–0.7), does a
> **process-aware (step-level) trained energy** + **optimal SC+energy aggregation**
> BEAT self-consistency at matched compute — the win GSM8K's ceiling structurally
> precluded? If it still ties/loses where SC has room, the energy-selection Phase-3
> premise is honestly REFUTED on this substrate (a strong citable negative). If it
> wins, it is the FIRST real Phase-3 justification.

This answers a question the GSM8K corpus **structurally cannot** (ceiling), uses a
**new technique** (per-step process energy + principled aggregation, not candidate-
level argmin), and is **tautology-clean by construction** (flip-count primary metric —
no two bit-identical accuracy fields). It is therefore depth, not breadth.

## 4. Phases

```
PHASE A  OPS transition
  exp3470  archive .319 / activate .320

PHASE B  DEPTH BLOCK (majority of slots; all on claude, heavy on opus)
  exp3471  HEADROOM corpus builder — hard-math cached corpus (SC ~0.4-0.7),
           resumable, per-STEP traces for process-reward scoring        [live GPU]
  exp3472  P0.1 v6 — process-aware step-level energy + trained EORM +
           optimal SC+energy aggregation vs SC on the HEADROOM corpus,
           flip-count primary (tautology-clean by construction)  [CRUX, cached]
  exp3473  calibration v3 — does the process energy track correctness AND
           recover minority-yet-correct answers on the headroom corpus? [cached]
  exp3474  FR-11 self-learning DEEPER — push the loop to N>=200 to test
           whether collapse emerges at depth; finally de-flag    [self-learning]
  exp3475  Kona on HARDER instances with headroom — does the trained/process
           energy lift the global-opt hybrid where instances aren't trivial?

PHASE C  G2 (the sole publication gate)
  exp3476  G2 self-contained, repo-independent reproduction package
           (tarball + pinned deps + one script + content-addressed CID),
           the "true-stranger" repro that needs zero Carnot knowledge

PHASE D  HARDWARE (light, opportunistic; north-star §3)
  exp3477  KV260 terminal latency transcript (SSH precondition)
  exp3478  GateMate opportunistic detect + toolchain continuity
  exp3479  PolarFire opportunistic reachability

PHASE E  OPS synthesis + capstone
  exp3480  G1-G4 gate-status synthesis v320
  exp3481  capstone v320
```

## 5. Dependency graph

```
exp3470 (archive/activate)
   │
exp3471 (headroom corpus, live GPU) ....... best-effort; NOT a hard gate
   │  (corpus on disk; exp3472 falls back to its own preconditions)
exp3472 (P0.1 v6 CRUX, cached) ── always emits a complete: verdict
   ├──> exp3473 (calibration v3)            [own preconditions, not hard-gated]
   ├──> exp3475 (Kona harder)               [gated_on exp3472 verdict contains complete]
   └──> exp3480 (gate synthesis)            [gated_on exp3472 verdict contains complete]
exp3474 (FR-11 deeper) ............ independent self-learning slot
exp3476 (G2 package) .............. independent
exp3477/3478/3479 (hardware) ...... independent, light
exp3480 (gate synth) ──> exp3481 (capstone)  [gated_on exp3480 gate_status_v320_ready]
```

**Cascade-safety (the .317 lesson):** the gate chain depends on the CHEAP cached
scoring crux (exp3472), never the live corpus builder (exp3471). exp3472 handles a
small/absent corpus via its own PRECONDITIONS and always emits a `complete:` verdict
(including a clean `complete: blocked_corpus_*`), so a CUDA-blocked exp3471 cannot
cascade-block the milestone.

## 6. Hardware requirements

- exp3471: CUDA (RTX 3090) + a cached SOTA GGUF (gemma-4-26B-A4B-it-GGUF, the fastest
  SOTA MoE, for corpus-homogeneity with the .318/.319 generation runs).
- exp3472/3473/3474/3475: cached/CPU only — no live model, cannot time out.
- exp3477: KV260 over SSH (`ssh kria`), honest `blocked_*` if unreachable.
- exp3478/3479: GateMate / PolarFire detect over USB-JTAG / SSH; honest `blocked_*` ok.

## 7. Model / agent routing

- **gemini-cli is DOWN** (.315/.316/.318 every gemini task crashed; no evidence of
  recovery). **ALL tasks route to `agent_type: claude`** with `requires_claude: true`
  (robust to `GEMINI_FORCE_EXPERIMENTS` / `CODEX_FORCE_EXPERIMENTS` coercion). Heavy /
  long / hardware tasks route to `model: opus` with `max_turns: 100`.
- SOTA model for the live corpus build: `unsloth/gemma-4-26B-A4B-it-GGUF` via the GGUF
  path (embedded tokenizer; NEVER `AutoTokenizer` on a `-GGUF` repo id — the 2026-05-29
  GGUF tokenizer rule).

## 8. Exit criteria / what "done" looks like

- **P0.1 v6 clean verdict** (un-flagged) on a headroom benchmark: either the process
  energy + aggregation BEATS SC (first real Phase-3 justification, preliminary until
  replicated) OR ties/loses with headroom present (premise refuted on this substrate —
  a strong, citable negative). Either way, the flip-count design must avoid the
  tautology flag.
- **G2** a repo-independent, one-command, content-addressed reproduction package
  exists; operator-action note surfaced (only a non-operator/CI run flips G2 → met).
- **FR-11** a clean (un-flagged) self-learning result at N≥200 depth.
- Capstone honestly reports which of G1–G4 moved and whether Depth-Over-Breadth can
  now relax (clean P0.1 verdict + in-flight G2 reproducer).

## 9. Discipline compliance

- **Depth-Over-Breadth:** majority of substantive slots on P0.1 (headroom) + G2; no
  vN+1 re-measurement of an already-measured artifact — exp3472/3473/3475 each answer
  a question the saturated GSM8K/Kona substrates structurally could not.
- **Hardware-Task Continuity / north-star §3:** KV260 (drive-to-terminal) + opportunistic
  GateMate/PolarFire, kept light.
- **Continuous self-learning:** exp3474 (mandatory).
- **Adversarial / tautology guards:** every depth task is tautology-clean by
  construction (flip-count / distinct-source metrics); flagged artifacts are excluded
  from aggregation.
- **Operator-Only External Publication:** exp3476 prepares the package but does NOT
  push or trigger external CI; G2-met requires operator action.
- **Verdict Terminal-Prefix / Principle-Annotated Fields / Pre-Launch Preconditions:**
  honored in every task prompt.
