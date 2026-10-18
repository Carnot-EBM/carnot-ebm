# Research Roadmap — Milestone 2026.06.387

**Title:** PROVE-OR-BOUND THE VERIFIER MOAT WITH HEADROOM CONTROLS — headroom-gated
executable moat test + V-STaR learned selector + GAP-3 Stage-1 model-native ARC energy

**Planned:** 2026-06-14 (UTC) · **Supersedes:** 2026.06.386 · **Planner:** Opus 4.8 (outer-loop)

---

## 0. One-paragraph thesis

The verifier is now Carnot's **entire, existential** value-add (north-star §5: energy
VERIFIES, refinement GENERATES). Milestone .386 ran the decisive verifier-as-reward graft on
the outer-loop-trained TRM Sudoku baseline (val 0.8227) and got a **NULL** —
`verifier_value_added: false`, rerank lift +0.0156 (CI95 [0, 0.047]), RFT delta +0.019
(CI95 [0, 0.058]). **But the null is uninformative:** oracle@k was 0.8125 ≈ the 0.82 baseline,
so Sudoku-Extreme at this checkpoint has **almost no selectable headroom** — the test cannot
distinguish "verifier fails" from "nothing to select" (CLAUDE.md FALSE_NEGATIVE_RISK). The
two corpora we have show the two failure modes cleanly: **Sudoku** (verifier executes perfectly,
zero headroom) and **ARC** (real ~13pp headroom: oracle 0.61 > vote 0.45, but the cheap
hand-invariant verifier *anti-ranks* — `ops/verifier_gaps.md` GAP-3). Fresh literature
(arXiv:2605.07395) adds a third hazard: apparent headroom can itself be an *evaluation artifact*.
So .387 answers the moat question **properly**: (A) build a headroom gate with an
**objective/executable oracle**, find an executable domain that genuinely *has* selectable
headroom, and run the moat test there with a **matched no-verifier control**; (B) build the
**GAP-3 Stage-1 model-native ARC energy** — the learned verifier that reaches the proven ~13pp
ARC headroom (the activation dump already exists, so this is CPU-only); (C) advance ARC solve
count monotonically; (D) reserved infra/hardware/SOTA/capstone. **DiffusionGemma stays GATED** —
it activates only when a headroom-present executable domain shows `verifier_value_added == true`.

## 1. What .386 proved (the inputs to this plan)

| Result | Artifact | Reading |
|---|---|---|
| Outer-loop TRM training converged to val **0.8227** (plateaued ~0.82, ~5pp shy of published 0.87) | `results/trm_runs/.../last.ckpt`, `contiguous_run.log` | Stable, faithful baseline; checkpoint frozen (train SIGTERM'd) |
| Decisive graft **FIRED, returned A≈B NULL** | exp4168 | `verifier_value_added: false`; but oracle@k 0.8125 ≈ baseline → **headroom-limited null** |
| **DiffusionGemma gate: STILL-PENDING** | exp4173 capstone | gate needs `verifier_value_added==true`; .386 was false → do NOT activate |
| ARC: `total_games_solved=13`, no solve this milestone — **non-spatial candidates exhausted** | exp4169 | remaining games spatial (HARD); need deeper-level or spatial target |
| SOTA ingestion flagged: **V-STaR rejected-trace selector + headroom gate BEFORE DiffusionGemma** | exp4170 | the explicit .387 directive from the loop's own ingestion |
| ARC hand-invariant verifier **anti-ranks** on TRM's real pool; **GAP-3 (learned energy) escalated to PRIMARY**; **Stage-0 q_halt scalar NEGATIVE**, Stage-1 latent is the GO | `ops/verifier_gaps.md`, `gap3-...-design.md` | the activation dump (`arc3_gap3_stage1_candidate_table.npz`, z_mean (8041,512)) already exists → Stage-1 is CPU-only |

## 2. The three biggest gaps (current state vs PRD vision)

1. **The moat is UNPROVEN AND UNBOUNDED.** The single existential question — does the external
   verifier beat the generator's own self-consistency, cheaper? — has only ever been tested on
   corpora that *structurally cannot answer it* (Sudoku: no headroom; ARC hand-invariants: wrong
   tool). .387 closes this with a positive-control (headroom-gated) executable moat test.
2. **The ARC verifier cannot reach its own measured headroom.** ~13pp of selectable headroom is
   proven to exist in TRM's ARC pool, and cheap hand-invariants are exhausted (refuted). The
   learned model-native energy (GAP-3 Stage-1) is the core-product build that would close it.
3. **No learned-selector arm exists.** Every verifier tested to date is a *hand-built* invariant
   or *outcome* checker. V-STaR (accepted+rejected traces) is the missing learned-selector class,
   and is the prerequisite gate the loop's own ingestion flagged for DiffusionGemma.

## 3. Architecture of the milestone

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  PHASE A — THE HEADROOM-CONTROLLED MOAT TEST (decisive)   │
                    │                                                           │
  cached candidate  │  A1 headroom gate ──► finds executable domain WITH        │
  pools (code/math/ │     + executable-oracle    selectable headroom (oracle@k  │
  Sudoku)           │       census               − baseline ≥ 0.10, artifact-   │
                    │                            sanitized per arXiv:2605.07395) │
                    │            │ max_selectable_headroom (BARE)                │
                    │            ▼ gated_on                                      │
                    │  A2 V-STaR ──────────► A3 DECISIVE MOAT TEST              │
                    │   learned selector      executable verifier + V-STaR vs   │
                    │   (accepted+rejected)   SC-vote vs oracle, MATCHED        │
                    │                         no-verifier control, accuracy+cost │
                    └───────────────────────────────┬─────────────────────────┘
                                                     │ verifier_value_added?
   ┌─────────────────────────────────────────┐      │      ┌──────────────────────────────┐
   │ PHASE B — GAP-3 STAGE-1 (CPU; dump done) │      │      │ PHASE C — ARC SOLVE (+1)     │
   │ model-native basis (1a) + learned probe  │      │      │ deeper level / spatial game  │
   │ (1b) on existing z_mean (8041,512); §3   │      │      │ explore→induce→verify;       │
   │ gates + §4 adversarial; reaches ~13pp?   │      │      │ monotonic, real-env-confirm  │
   └─────────────────────────────────────────┘      │      └──────────────────────────────┘
                                                     ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │ PHASE D — RESERVED SLOTS: D1 SOTA-ingestion · D2 registry+gaps hygiene · D3 hardware · │
   │ D4 capstone (does the moat resolve? is the DiffusionGemma gate now MET?)               │
   └──────────────────────────────────────────────────────────────────────────────────────┘
```

## 4. Phase descriptions

### Phase A — The headroom-controlled moat test (the decisive question, done right)

The .386 null was headroom-limited. The fix is a **positive control**: only test the moat where
selectable headroom genuinely exists, measured with an **objective/executable** oracle (not an
LLM judge — arXiv:2605.07395), and always against a **matched no-verifier control** (TTA-TRM
arXiv:2511.02886: a tiny refiner can improve from adaptation compute alone, so any verifier win
must beat same-budget no-verifier). Report **accuracy AND cost** (arXiv:2504.01005: SC is
compute-cheaper than a generative verifier until ~8×; Carnot's win condition is
efficiency-parity, north-star §5).

- **A1 — Headroom gate + executable-oracle corpus census** (`exp4175`). Build reusable
  `scripts/headroom_gate.py`: for a candidate pool, compute oracle@k vs baseline pass@1 vs
  SC-vote with an EXECUTABLE oracle, and flag artifact inflation (truncation/format). Census the
  cached pools (code: HumanEval/MBPP from exp1999/2090/1607; math: GSM8K; Sudoku held-out). Emit
  `max_selectable_headroom` (BARE float) + the chosen `headroom_present_domain`. **The positive
  control the moat test requires.**
- **A2 — V-STaR learned selector** (`exp4176`). The missing learned-selector class (arXiv:2402.06457):
  train a selector on **accepted AND rejected** candidate traces (oracle-free, OOF), so it learns
  a correctness boundary rather than imitating winners. Deployable ranker for A3.
- **A3 — THE decisive headroom-controlled moat test** (`exp4177`, gated_on A1). On the
  headroom-present executable domain: executable verifier ensemble + V-STaR (A2) vs SC-vote vs
  oracle, **matched no-verifier control**, accuracy + cost. Decisive: does the verifier add value
  (CI95 excl 0) **where headroom exists**? This is the moat-lineage continuation that addresses
  the exp4168 headroom-limited null.

### Phase B — GAP-3 Stage-1: the learned model-native ARC energy (core-product verifier build)

`ops/verifier_gaps.md` GAP-3 is escalated to PRIMARY: cheap hand-invariants are exhausted, only a
content/rule-aware *learned* energy can reach the proven ~13pp headroom. Stage-0 (q_halt scalar)
was NEGATIVE but the 0.86 within-task soft-AUROC says the signal is in the *latent* → Stage-1 is
the GO. **The activation dump already exists** (`arc3_gap3_stage1_candidate_table.npz`:
z_mean (8041,512), votes, q_mean, probe, correct, task_idx) — so Stage-1 is **CPU-only**, no GPU,
no collision with training.

- **B — GAP-3 Stage-1 model-native energy** (`exp4178`). On the existing npz: (1a) recover an
  orthogonal **model-native basis** (arXiv:2604.17614) from `z_mean` and score candidate
  consistency; (1b) the **learned probe** energy (OOF). Run ALL §3 gates (selection > vote,
  AUROC > 0.70, coverage ≥ 80%, headroom-capture ≥ 30%) + ALL §4 adversarial checks (permutation
  control, strict OOF, oracle-leak audit, bootstrap CI). Honest negative is a COMPLETE verdict.

### Phase C — ARC incremental progress (monotonic +1)

- **C — ARC incremental +1** (`exp4179`). Non-spatial first-levels are exhausted, so target the
  **next deeper level of an already-solved game** (incremental +1 per the ARC Incremental-Progress
  Scoping discipline) via explore→induce→verify, or a spatial game's L1. Real-env-confirm; STOP at
  the first level that fails. Honest no-solve = COMPLETE.

### Phase D — Reserved slots (infra / hardware / SOTA / capstone)

- **D1 — SOTA-ingestion** (`exp4180`, reserved bleeding-edge slot). Ingest the .387 sweep
  (2605.07395, 2504.01005, 2510.20607, 2602.01849, 2512.11847) mapped onto the moat / GAP-3 /
  DiffusionGemma stack; flag the strongest for .388. Real arXiv IDs mandatory.
- **D2 — Verifier-registry + gaps hygiene** (`exp4181`, reserved infra slot). Bit-exact GAP-4
  ARC-1 regression replay; record the .387 moat verdict + GAP-3 Stage-1 result into
  `ops/verifier_gaps.md` (never-prune).
- **D3 — Hardware continuity** (`exp4182`). GateMate + PolarFire drive-toward-terminal; KV260
  opportunistic. SSH/USB-detect preconditions ONLY (KV260 SSH-Not-SD-Card Discipline).
- **D4 — Capstone .387** (`exp4183`). Headline question: **did the headroom-controlled moat test
  resolve the verifier-value question (positive where headroom exists, or honestly bounded), and
  did GAP-3 Stage-1 reach the ~13pp ARC headroom?** Resolve whether the **DiffusionGemma gate is
  now MET**. SKIP any `flagged_adversarial` artifact; cite upstream sha256.

## 5. The DiffusionGemma gate (explicitly respected)

DiffusionGemma energy-guided diffusion is SPECCED and GATED
(`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md`): activate ONLY after a
verifier-graft reports `verifier_value_added == true` on an executable domain. The .386 Sudoku
graft was NULL **because Sudoku has no headroom** — a headroom-dead domain can never flip the
gate, which would trap the scale-up forever. **.387's reframing (encoded in the capstone):** the
gate's real intent is "the verifier adds value on an executable domain *with selectable
headroom*." If A3 shows `verifier_value_added==true` on the headroom-present executable domain
(matched control, CI95 excl 0), the gate is MET and .388 may activate DiffusionGemma. If A3 is a
clean (headroom-present) null, the gate stays closed and the bottleneck is verifier
discrimination — exactly the GAP-3 work. **.387 does NOT launch DiffusionGemma.**

## 6. Dependency graph

```
exp4174 (archive/activate) ─► exp4175 (A1 headroom gate) ─► exp4177 (A3 moat test, gated_on A1)
                              exp4176 (A2 V-STaR) ──────────►   ▲ (selector input)
                              exp4178 (B GAP-3 Stage-1) ── independent (CPU, npz exists)
                              exp4179 (C ARC +1) ───────────── independent
                              exp4180 (D1 SOTA) ─┐
                              exp4181 (D2 registry) ─┤─► exp4183 (D4 capstone, ungated, aggregates all)
                              exp4182 (D3 hardware) ─┘
```

## 7. Hardware requirements

- **No GPU strictly required this milestone.** Phase B is CPU (npz exists). Phase A prefers cached
  candidate pools; A3 may generate fresh candidates from a SOTA GGUF (`gemma-4-26B-A4B-it-GGUF` /
  `Qwen3.6-35B-A3B-GGUF`) behind a PRECONDITIONS cache check — declared, not assumed.
- **No write to the TRM stable checkpoint dir** (read-only; avoid the .385 collision). Training is
  done (SIGTERM'd at val 0.8227); the conductor remains stood-down on TRM training.
- **FPGA boards** (D3): GateMate (USB-detect), PolarFire (ssh), KV260 (ssh, terminal/opportunistic).

## 8. Self-learning coverage (research-program.md requirement)

Two learned/self-improving verifiers this milestone: **A2 V-STaR selector** (learns a correctness
boundary from accepted+rejected traces) and **B GAP-3 learned probe energy** (Tier-3 predictive
verification — learns to predict correctness from the generator's own latent). Both advance the
continuous-self-learning architecture (FR-11 / Tier-3).

## 9. Disciplines honored

Codex-Default v2 (all tasks codex/gpt-5.5) · Verdict Terminal-Prefix · Principle-Annotated fields +
gates · Pre-Launch Preconditions · Inference-Substrate Declaration · FALSE_NEGATIVE_RISK /
positive-control (the entire Phase-A design) · Reading-Results Discipline · Exclusion-Manifest
cross-check (no task matches the 15 retired ids) · `operator_override` on every lineage-continuation
task (auto-override classes 1/2/3) · ARC Incremental-Progress Scoping · SOTA-Ingestion Cycle ·
Missing-Verifier Gap Logging · Hardware-Task Continuity · Reserved infra slots (≥2) · Operator-Only
External Publication (no submission steps).
