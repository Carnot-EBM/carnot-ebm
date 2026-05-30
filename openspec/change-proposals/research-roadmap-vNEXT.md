# Research Roadmap — Milestone 2026.05.321 (Depth-Over-Breadth VII)

**Title:** P0.1 on a Difficulty-Matched Headroom Corpus (Fix the SC-Band Precondition)
+ FR-11 Minimal-β & Grounding-Dependence + Kona Harder Instances + G2 External-Ask

**Planner:** Claude Opus 4.8, 2026-05-30 (Depth-Over-Breadth Forcing Function ACTIVE).
**Milestone doc format:** follows v7/v8.
**Prior milestone:** 2026.05.320 (Depth-Over-Breadth VI).

---

## 1. What the previous milestone (.320) proved

`.320` pivoted P0.1 (does energy-based selection beat self-consistency at equal
compute?) off the **saturated GSM8K ceiling** (SC 0.908, where .319 found an exact
tie) onto a **HEADROOM** benchmark. The result is sharp but **blocked on a
benchmark-selection precondition — not on the energy substrate**:

| Exp | Verdict | Outcome |
|---|---|---|
| exp3471 corpus builder | `blocked_no_headroom` | MATH **Level 5**, Gemma4-26B, k=6 → **SC 0.265**, below band floor [0.40] |
| exp3472 P0.1 v6 (crux) | `blocked_p01_corpus_too_small_n=21` | **No energy-vs-SC comparison run. P0.1 REMAINS OPEN.** |
| exp3473 calibration v3 | FLAGGED (TAUTOLOGY) | Advisory: process AUROC **0.441 (below chance)** on MATH → FoVer ensemble domain-specificity concern |
| **exp3474 FR-11 depth** | **CLEAN (de-flagged)** | **Collapse at N=200 (onset 138); `entropy_beta=0.50` cures it.** Depth-sensitive. |
| exp3475 Kona | `blocked_saturated` | Untrained hybrid solve-rate **1.0** — no headroom (GSM8K-ceiling analogue) |
| exp3476 G2 package | CLEAN | Self-contained tarball + SHA256 + IPFS CID, AUROC 0.9131 verified; **external run pending** |

Hardware: KV260 `blocked_ssh_unreachable`, GateMate `blocked_toolchain_missing`,
PolarFire reachable (continuity confirmed).

**Gate status:** G1 ✓, G3 ✓, G4 ✓, **G2 ✗** (sole unmet). `paper_ready = false`.
**Depth-Over-Breadth Forcing Function REMAINS ACTIVE** (P0.1 has no clean verdict;
G2 has no confirmed external run).

**The single load-bearing diagnosis:** the .320 corpus used **MATH Level 5**, too hard
for Gemma4-26B (SC 0.265 ≪ band floor). The selection premise was never tested because
the precondition (a non-degenerate majority vote with room to improve) was never met.
The corpus builder already has all the per-step-trace machinery; **only the benchmark
difficulty needs to change.**

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **P0.1 is still OPEN** — the single most important test (energy-based selection vs
   self-consistency, the Phase-3 non-autoregressive-reasoning premise) has never had a
   clean run because both attempted substrates were degenerate (GSM8K at ceiling 0.908;
   MATH Level 5 at floor 0.265). **Gap: a difficulty-matched corpus where SC ∈ [0.40,
   0.70].** Literature (ThinkPRM, arXiv:2504.16828) locates this at **MATH-500 levels
   3-4**.
2. **The verifier ensemble's domain transfer is unmeasured** — exp3473's advisory
   (process AUROC 0.441 < chance on MATH) suggests the FoVer 4-verifier ensemble,
   designed for GSM8K/FoVer, may not discriminate correctness on MATH. If the energy
   carries no signal on the test corpus, P0.1 cannot be answered regardless of headroom.
   **Gap: a clean (de-flagged) MATH-domain energy calibration.**
3. **G2 (independent reproduction) is unclosable by the autonomous loop** — the package
   is built and internally verified; closing it requires a **non-operator external run**
   (Operator-Only External Publication). **Gap: the lowest-friction external-ask** — a
   public one-click `workflow_dispatch` path + a regression-verified package, so the
   operator's action is a single click.

---

## 3. Architecture diagram (the P0.1 difficulty-matched pipeline)

```
                       .321 P0.1 v7 PIPELINE (the fix)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ exp3483  HEADROOM CORPUS BUILDER v2  (live GGUF, GPU)                   │
  │   MATH-500 levels 3-4  ──ADAPTIVE warm-up SC probe──> select in-band    │
  │   split where SC ∈ [0.40, 0.70]   (fallback: stronger SOTA / higher k)  │
  │   per problem: 1 greedy + k=6 sampled + per-STEP traces + logprobs      │
  │           │  data/p01_difficulty_matched_generations.jsonl              │
  └───────────┼────────────────────────────────────────────────────────────┘
              │ (cached; downstream NOT gated on the live builder —
              │  each consumer handles small/absent corpus via its own
              │  PRECONDITIONS → clean blocked verdict, no cascade)
   ┌──────────┴───────────┐        ┌─────────────────────────────┐
   │ exp3484  P0.1 v7 CRUX │        │ exp3485  CALIBRATION v4      │
   │ process-step energy + │        │ (de-flagged): process AUROC  │
   │ optimal SC+energy agg │◀──────▶│ + minority-correct recovery  │
   │ vs SC @ matched compute│  same  │ on MATH; MATH-aware recal.   │
   │ FLIP-COUNT (taut-clean)│  corpus│ distinct energies (asserted) │
   └──────────┬───────────┘        └─────────────────────────────┘
              │ honest_verdict (cheap, reliable — the gate-chain anchor)
   ┌──────────┴────────────────────────────────────────────────────────────┐
   │ exp3492  G1-G4 SYNTHESIS  ──gated_on exp3484.honest_verdict contains    │
   │ 'complete'──>  exp3493  CAPSTONE                                         │
   └─────────────────────────────────────────────────────────────────────────┘

  PARALLEL DEPTH:  exp3486 FR-11 minimal-β + grounding-dependence (mandatory
                   self-learning) │ exp3487 Kona harder-instance generation
  G2:              exp3488 public one-click external-ask + regression-verify
  HARDWARE:        exp3489 KV260 terminal │ exp3490 GateMate │ exp3491 PolarFire
```

**Gate-chain principle (the .317 cascade lesson):** the synthesis/capstone gate on the
**cheap, reliable CACHED scoring crux (exp3484)**, never the live generation task
(exp3483). exp3484 emits a `complete:` verdict under EVERY outcome (including a clean
`blocked_corpus_too_small` / `blocked_corpus_at_ceiling`), so a GPU-blocked builder
cannot cascade-skip the gate chain.

---

## 4. Phases

### Phase A — OPS transition (1 task)
- **exp3482** archive .320, write the operational retro, activate .321.

### Phase B — DEPTH BLOCK (5 tasks; the majority of substantive slots)
- **exp3483** P0.1 HEADROOM corpus builder v2 — **difficulty-matched** (MATH-500 levels
  3-4) with an adaptive warm-up that selects the split where SC ∈ [0.40, 0.70]. The ONE
  change from exp3471: benchmark difficulty. Live GGUF, GPU, resumable, timeout-proof.
- **exp3484** P0.1 v7 — process-aware step-level energy + optimal SC+energy aggregation
  vs self-consistency at matched compute, flip-count tautology-clean. THE crux. Cached.
- **exp3485** calibration v4 (de-flagged) — process & trained energy AUROC + minority-
  correct recovery on the in-band corpus, with MATH-aware recalibration; fixes the
  exp3473 tautology by computing the two energies from genuinely distinct pipelines
  (runtime assert). Cached.
- **exp3486** FR-11 minimal-β + grounding-dependence (**mandatory continuous-self-
  learning task**) — exp3474 showed β=0.50 cures collapse at N=200; this finds the
  MINIMAL effective β (sweep {0, 0.1, 0.25, 0.5}) and whether collapse onset moves with
  grounding strength (ACTIVE_WEIGHT). Cites ER-PRM. Cached.
- **exp3487** Kona harder-instance generation + process hybrid — exp3475 was saturated;
  this generates harder instances (untrained-hybrid solve-rate < 0.8) behind an
  encoding-validity gate (E==0 on a known-valid board) per the known-issues KONA
  correctness-first gate, then tests the process energy as proposal heuristic. Cached/CPU.

### Phase C — G2 (1 task; the sole publication gate)
- **exp3488** FoVer G2 public one-click external-ask + package regression-verify —
  regression-verify the exp3476 package still reproduces from a clean env, and prepare
  the lowest-friction external ask (public `workflow_dispatch` workflow + reproducer
  invite + operator checklist). Does NOT push, does NOT trigger CI, does NOT mark G2 met.

### Phase D — HARDWARE (3 tasks; light, opportunistic per north-star §3)
- **exp3489** KV260 terminal latency transcript (re-attempt SSH; mandatory until terminal).
- **exp3490** GateMate opportunistic detect + toolchain continuity (no flash mandate).
- **exp3491** PolarFire opportunistic reachability audit (no terminal mandate).

### Phase E — OPS synthesis + capstone (2 tasks)
- **exp3492** G1-G4 gate-status synthesis (gated on the cheap exp3484 crux).
- **exp3493** Capstone v321 (gated on exp3492).

---

## 5. Dependency graph

```
exp3482 (archive/activate)  ──> [all]
exp3483 (corpus builder)    ──(cached file; soft)──> exp3484, exp3485, exp3487
exp3484 (P0.1 crux)         ──gated_on honest_verdict contains 'complete'──> exp3492
exp3485, exp3486, exp3487   ──(read by)──> exp3492
exp3488 (G2), exp3489-91 (hw)──(read by)──> exp3492
exp3492 (synthesis)         ──gated_on gate_status_v321_ready==true──> exp3493 (capstone)
```

All downstream depth consumers handle a small/absent corpus via their own PRECONDITIONS
(clean `blocked_*` verdicts), so a CUDA-blocked exp3483 cannot cascade. exp3487 is
additionally `gated_on exp3484.honest_verdict contains 'complete'` (the process energy
must exist before it is reused as a Kona heuristic).

---

## 6. Hardware requirements

| Task | Hardware | Substrate |
|---|---|---|
| exp3483 corpus builder | 1× RTX 3090 (CUDA) + cached SOTA GGUF | `live_llm_inference` |
| exp3484/3485/3486/3487 | CPU only (cached scoring) | `verifier_ensemble_against_cached_candidates` |
| exp3488 G2 | CPU (Docker/fresh venv) | `verifier_ensemble_against_cached_candidates` |
| exp3489 KV260 | SSH to `kria` board | `hardware_smoke` |
| exp3490 GateMate | DirtyJtag USB + himbaechel toolchain | `hardware_smoke` |
| exp3491 PolarFire | SSH to `polarfire` board | `hardware_smoke` |
| exp3482/3492/3493 | CPU (aggregation) | `aggregation_from_upstream_artifacts` |

**Models:** SOTA GGUF per CLAUDE.md — `unsloth/gemma-4-26B-A4B-it-GGUF` (default,
homogeneous with the .320 corpus), with `unsloth/gemma-4-31B-it-GGUF` as the stronger
fallback if even MATH Level 3 lands below the band. GGUF path only (embedded tokenizer;
never `AutoTokenizer` on a `-GGUF` repo id per the 2026-05-29 GGUF tokenizer rule).

---

## 7. Continuous self-learning coverage (PRD requirement)

**exp3486** is the mandatory continuous-self-learning task. It advances the FR-11 /
Tier-3 self-learning track by determining the **minimal entropy regularization** that
keeps the self-distillation loop from collapsing onto the verifier null space (the
Q12 Dark-Room / Zenil α_t failure mode), and whether the collapse onset depends on the
grounding diversity. This is a genuine forward step (minimal-β + grounding-dependence),
not a re-measurement of exp3474's N=200 confirmation — it converts "β=0.50 works" into
"β≥X is necessary and sufficient, and the safety margin scales with grounding," the
actionable Phase-5 pre-deployment default.

---

## 8. Discipline compliance checklist

- **Depth-Over-Breadth:** 5 of 12 tasks are P0.1/Kona/self-learning depth; 1 is G2 (the
  finish line). No `vN+1` re-measurement of an already-measured artifact — every depth
  task answers a question its predecessor structurally could not (in-band corpus,
  MATH-domain calibration, minimal-β, harder Kona instances).
- **Hardware-Task Continuity:** one task per attached board (KV260 mandatory until
  terminal; GateMate + PolarFire opportunistic per north-star §3).
- **Gemini-cli is DOWN** (.315/.316/.318 crashed every gemini task) → ALL tasks
  `agent_type: claude, requires_claude: true`; heavy/long/hardware on `model: opus`.
- **Pre-Launch Preconditions** on every compute-bound task; **KV260 SSH-not-SD-card**;
  **Verdict Terminal-Prefix** (`complete:`); **Inference-Substrate Declaration**;
  **Principle-Annotated** artifact fields + gates; **Adversarial / Sample-Size Rigor**
  (flip-count tautology-clean by construction; distinct-energy runtime asserts).
- **Exclusion-Manifest / Failed-Experiment Rerun:** every scope-matching task carries an
  `operator_override:` (standing 2026-05-29/30 directives) + a full `prior_failures:`
  block (all four sub-fields).
- **Operator-Only External Publication:** exp3488 prepares the external ask but never
  pushes, triggers CI, or marks G2 met.
- **Public Documentation Discipline:** no autonomous edits to north-star.md, index.html,
  README, or operator-curated docs.
