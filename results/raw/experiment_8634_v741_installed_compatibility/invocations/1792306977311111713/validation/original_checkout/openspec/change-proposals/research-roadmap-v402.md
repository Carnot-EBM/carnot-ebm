# Research Roadmap v402 — Convert the PROVEN oracle-distinct moat into a generation gain (S3) + drive ARC north-star solves DEEPER

**Milestone:** 2026.06.402
**Planned:** 2026-06-17 (UTC)
**Supersedes:** v401 (2026.06.401)
**North star:** `ops/north-star.md` §0 (solve ARC-AGI-3, accurately + efficiently), §1 (the FoVer headline / no-churn rule), §5 (energy VERIFIES, refinement GENERATES — the verifier-moat is existential)

---

## 1. What .401 PROVED (the true close-state)

.401 was DEPTH on the diagnosed .400 root causes, and — for the first time in the
DiffusionGemma arc — the headline bet PAID OFF.

| Axis | .401 result | Source |
|---|---|---|
| **HEADLINE — in-generation oracle-distinct moat** | **REPLICATED, leak-robust → DiffusionGemma gate MET.** A DiNa-LRM leak-robust partial-state reward scorer (exp4337: masked-answer recovery AUROC 0.56 ≈ chance, process-ranking AUROC 0.70 — oracle-distinct under answer-cell masking on ≥2 corpora) drove a clean 2nd-corpus replication: Carnot reward-guided step-stitching beat the best engaged control by **+0.358** and the intrinsic self-reward SMC by **+0.321**, **CI95 [0.283, 0.4375] EXCLUDES 0**, `controls_differentiated=true`, `scorer_leak_recheck_passed=true`, **n=240**, `verifier_is_oracle=false`. | exp4337, exp4338 |
| **ARC NORTH STAR — first E3 solves** | **ar25 L1 reproduced** (explore-verify-plan, AERA 2605.25931 + Agent2World 2512.22336) and **sc25 L1 reproduced** (the E3 induced the spell-cast/cast-grid mechanic the BFS solver stalled on — opening the path to sc25's 5 live-recorded levels). | exp4339, exp4341 |
| ARC — deep tail (still partial) | ka59 0.56, tr87/ft09 partial — graph-explore-resistant, mechanic-limited; the E3 induction is the right tool but did not yet reach a verified solve. | exp4340, exp4329 |
| ARC — outer-loop | tn36 program-editor RE: **L1–L6 reproduced** (multi-run maze path-planner). lp85 L3→L4. E1-sweep + adapter-free sweep added cd82/sp80/su15/tu93/cn04/m0r0/sk48. | `ops/arc_solve_registry.yaml` |
| **ARC reproducible scorecard** | **21 reproducible levels / 13 games** (registry, authoritative 2026-06-17). First live leaderboard submission: 13 levels, 11/11 env-matched, operator-gated. | registry; `results/arc3_live_submit.json` |
| SELF-LEARNING — cross-game transfer | **NULL again (3rd powered null) → RETIRED.** The action-role encoder value head did not transfer (reduction CI95 [1.0, 1.017], positive control passed). Now on `ops/exclusion_manifest.yaml`. | exp4342 |
| Publication gate | `paper_ready=True` — FoVer 0.9131, G1∧G2∧G3∧G4 all pass (stable). | `publication_gate.py` |

**The single most important fact:** the project's **first leak-robust, replicated,
oracle-distinct verifier-moat win** now exists. The verifier is no longer only
useful where it can *execute* (circular); a *learned* reward signal that is provably
NOT the oracle (answer-cell-masked) captures real in-generation headroom. The
DiffusionGemma gate — STILL-PENDING since 2026-06-13 — is **MET**.

---

## 2. The .402 thesis — from "the moat EXISTS" to "the moat DOES something"

A proven moat is a *measurement*; the north-star §5 win condition is *utility*: the
verifier earns its place when it makes generation **more accurate at equal compute,
or equally accurate at lower compute** (Pareto-dominate the base generator). .401
proved the oracle-distinct scorer ranks partial states correctly. .402 asks the
forward question: **does putting that scorer INSIDE the denoising loop produce a
better generation at a FIXED compute budget?**

The SOTA-ingestion slot (exp4343, all IDs WebFetch-verified 2026-06-17) flagged the
exact instrument: **S³ — Stratified Scaling Search (arXiv:2604.06260)**, a classical
verifier-guided search that, at each denoising step, expands candidate trajectories,
scores them with a lightweight reference-free verifier, and resamples the promising
ones while preserving frontier diversity — approximating a reward-tilted sampling
distribution anchored to the model prior. The Carnot leak-robust scorer (exp4337) IS
that verifier. The must-beat baseline is **best-of-K at matched NFE** (the same
compute spent on independent samples + final selection) and **self-reward SMC**
(intrinsic confidence — the sharpest oracle-distinct test). If S³-with-Carnot beats
both at fixed NFE, the oracle-distinct verifier Pareto-improves generation — the
headline graduates from "exists" to "useful."

Simultaneously, the operator's **2026-06-17 ARC MANDATE** continues: drive the E3
executable-world-model solver onto the still-partial deep tail (ka59, tr87, ft09) and
**DEEPER on the games we cracked** (sc25 L2+, the single biggest reproducible-level
upside at +4; ar25 L2; tn36 L7), monotonically raising the reproducible-level count.

```
                         CARNOT .402 ARCHITECTURE (hybrid: energy VERIFIES, refinement/LLM GENERATES)

  ┌─────────────────────────── HEADLINE: in-generation moat → generation gain ──────────────────────────┐
  │                                                                                                       │
  │   DiffusionGemma (Apache-2.0 26B/4B, llama.cpp PR binary) ── denoising trajectory ──►                 │
  │        │                                                                                              │
  │        ▼  at each step, expand K candidate partial states                                            │
  │   ┌──────────────────────────────┐     score (oracle-DISTINCT, answer-cell-masked)                   │
  │   │ DiNa-LRM leak-robust scorer   │ ◄── exp4337 .pkl (masked-answer AUROC 0.56 ≈ chance)             │
  │   │  (timestep-conditioned head)  │                                                                   │
  │   └──────────────┬───────────────┘                                                                   │
  │                  ▼  S³ stratified resample (keep promising, preserve diversity)                       │
  │           reward-tilted generation  ──►  vs best-of-K@NFE / unguided / self-reward-SMC               │
  │                  │                        (PAPO reward-state-alignment diagnostic gates the claim)    │
  └──────────────────┴────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────── NORTH STAR: ARC-AGI-3 (accuracy + efficiency) ───────────────────────────┐
  │   codex/gpt-5.5 PROPOSER ── induces a Python world model ──► Carnot WorldModelVerifier (the moat:    │
  │   collect_transitions → write engine() → score() → refactor → plan_and_execute → reproduce() gate)    │
  │   ka59 (closest new)   sc25 L2+ / ar25 L2 / tn36 L7 (deeper)   tr87+ft09 (breadth)                    │
  │                                                                                                       │
  │   SELF-LEARNING (compounds): learned A* action-cost heuristic ── trained on solved-game traces ──►    │
  │   action-MINIMAL replans (north-star EFFICIENCY axis: fewer env actions per solve, RHAE)              │
  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phases

### PHASE 0 — TRANSITION (exp4347)
Archive .401 → activate .402; record the TRUE .401 close-state (moat MET leak-robust;
ARC 21 reproducible / 13 games; ar25+sc25 E3-solved; ka59/tr87/ft09 still partial;
cross-game transfer RETIRED 3rd-null; `paper_ready=True`). codex (mechanical).

### PHASE A — HEADLINE: convert the proven moat into a generation gain
- **exp4348 (S³ stratified verifier-guided search, arXiv:2604.06260).** Put the
  exp4337 leak-robust scorer inside DiffusionGemma's denoising loop as the in-loop
  verifier. At a **FIXED NFE budget**, measure S³-Carnot vs best-of-K@NFE (compute-
  matched) vs unguided vs self-reward-SMC (intrinsic). Reuse the .401 skeptic-proof
  harness (no-op `controls_differentiated` guard; independent leak re-check). Emit
  `s3_guided_beats_control` BARE bool := (S³-Carnot − best-of-K > 0 AND CI95-excl-0
  AND controls_differentiated AND S³-Carnot − self-reward-SMC > 0). `verifier_is_oracle=false`.
  An honest null (the proven scorer does NOT convert to a fixed-NFE generation gain)
  is decision-grade.
- **exp4349 (PAPO reward-state-alignment diagnostic, arXiv:2606.08501) — GATED on
  exp4348 `s3_guided_beats_control==true`.** The operator was twice-burned on
  DiffusionGemma over-claims, so any positive gain must survive the "rewards become
  dense but wrong" failure mode: verify the S³ gain comes from rewards aligned to
  AUTHENTIC denoising states (not artifacts of remasking / position cues). Distinguish
  authentic trajectory states from random remasking; report alignment AUROC. If the
  gain is alignment-driven → the headline graduates; if it is an artifact → the gain
  is quarantined. `verifier_is_oracle=false`.

### PHASE B — ARC NORTH STAR (operator MANDATORY 2026-06-17; incremental-progress)
- **exp4350 (E3 ka59 → L1).** The closest still-partial deep-tail (0.56). More
  explore-verify-plan rounds in a quieter codex window. `verifier_is_oracle=true`.
- **exp4351 (E3 DEEPER on solved games).** sc25 L2+ (reproduce the live-recorded
  levels — biggest reproducible-level upside, +4) + ar25 L2 + tn36 L7, +1 each per
  the Incremental-Progress Scoping discipline. `verifier_is_oracle=true`.
- **exp4352 (E3 tr87 + ft09, +1 each, looped).** Breadth-of-progress on the two
  remaining partial deep-tail games; checkpoint per game. `verifier_is_oracle=true`.

### PHASE C — CONTINUOUS SELF-LEARNING (mandated; the WORKING, compounding mechanism)
- **exp4353 (learned A* action-cost heuristic for ARC action-efficiency).**
  Cross-game *value-transfer* is RETIRED (3 powered nulls: exp4318/4331/4342, on the
  exclusion manifest — generic / learned-frame / action-role features all failed to
  transfer a search-value signal). The self-learning mandate PIVOTS to the mechanism
  the registry shows WORKS and COMPOUNDS: the per-game learned verifier (lp85: 9.36×
  fewer states on held-out L3). .402 advances it from a steps-to-go *value head* to a
  learned **path-cost heuristic** that produces ACTION-MINIMAL plans — directly serving
  the north-star EFFICIENCY axis (fewer env actions per solve / lower RHAE), not just
  search compute. Train on the 21 solved-level traces; measure actions-to-solve
  reduction vs the current BFS/region-count plans on held-out levels, reproduction-gated.
  `verifier_is_oracle=false`.

### PHASE E — INFRA + HYGIENE + CAPSTONE
- **exp4354 (SOTA-ingestion → .403).** Mandatory per SOTA-Ingestion Cycle Discipline;
  reliable channel only (`sweep_*.py` + low-concurrency WebSearch/WebFetch; NO
  `/deep-research` in-loop). Flag A2D2/SEPO verifier-as-reward GENERATOR-training as
  OUT-OF-BAND/operator-owned (NOT auto-run). Emit `flagged_for_v403`.
- **exp4355 (registry/gaps hygiene + GAP-4 regression guard + capstone-stamping fix).**
  Also fix the `.401 capstone CIRCULAR_MOAT_OVERCLAIM stamp (exp4346 omitted
  `verifier_is_oracle` → flagged; the underlying exp4338 is correctly `false`).
  Ensure the .402 capstone carries `verifier_is_oracle` per the Circularity Discipline.
- **exp4356 (hardware continuity — KV260 opportunistic).** Per north-star §3, KV260 is
  THE sovereignty story; opportunistic SSH-reachability + bitstream check. NEVER a host
  SD-card precondition (KV260 SSH-Not-SD-Card Discipline).
- **exp4357 (capstone .402).** The headline decision (did S³ convert the moat to a
  fixed-NFE generation gain?) + ARC reproducible-total + the cross-game-transfer
  retirement + G1–G4. opus.

---

## 4. Dependency graph

```
exp4347 (archive/activate)
   │
   ├─► PHASE A ─ exp4348 (S³ search) ──gated──► exp4349 (PAPO alignment diagnostic)
   │
   ├─► PHASE B ─ exp4350 (ka59) │ exp4351 (sc25 L2+/ar25 L2/tn36 L7) │ exp4352 (tr87+ft09)   [independent]
   │
   ├─► PHASE C ─ exp4353 (learned action-cost heuristic)                                       [independent]
   │
   └─► PHASE E ─ exp4354 (SOTA→.403) │ exp4355 (hygiene) │ exp4356 (KV260)
                    │                       │                    │
                    └───────────────────────┴────────────────────┴──► exp4357 (capstone .402)
```

Only one hard gate (exp4349 ⟸ exp4348). PHASE B/C tasks are independent (no cross-gating)
so a single slow E3 run never blocks the others. The capstone reads all upstream artifacts.

---

## 5. Hardware requirements

| Task | Hardware | Substrate |
|---|---|---|
| exp4348 / exp4349 (S³ + PAPO) | 1× RTX 3090 (DiffusionGemma Q4_K_M, 16 GB, llama.cpp PR binary) | `live_llm_inference` |
| exp4350 / exp4351 / exp4352 (E3 ARC) | codex/gpt-5.5 proposer; offline ARC sim (`environment_files/`, zero quota) | `live_llm_inference` |
| exp4353 (action-cost heuristic) | CPU (offline solve traces + search) | `aggregation_from_upstream_artifacts` |
| exp4356 (KV260) | KV260 via SSH (opportunistic) | `hardware_smoke` |
| transition / hygiene / SOTA / capstone | CPU | `aggregation_from_upstream_artifacts` |

---

## 6. HARD RULES (every task — carried forward from .401, unchanged)

1. **Conductor STOOD-DOWN on TRM training.** No task launches TRM training, runs
   `pkill`/`kill` against `train.py`, or writes `results/trm_runs/`.
2. **Qwen FORBIDDEN as the TRAINED base** (Spurious-Rewards confound). Qwen/Gemma GGUF
   as an off-policy JUDGE/GENERATOR is fine.
3. **SEPO / A2D2 (verifier-as-reward GENERATOR training, 2502.01384 / 2606.13565) is
   OUT-OF-BAND / operator-owned** — flagged in SOTA-ingestion, NOT auto-run in-loop.
   The S³ headline (exp4348) is a NO-TRAINING test-time search (no weight updates).
4. **Circularity Discipline:** every learned-verifier value task declares
   `verifier_is_oracle` honestly. An EXECUTABLE-oracle win (an E3 solve) is
   `execution_grounded` (`verifier_is_oracle=true`, ARC progress, NOT a moat headline);
   the moat is the oracle-DISTINCT learned result with a matched control + CI95-excl-0.
5. **DiffusionGemma via the llama.cpp PR binary** (`~/.cache/llama.cpp-master/build/bin/
   llama-diffusion-gemma-eval`), NOT a standard GGUF loader (known-issues 2026-06-15).
6. **NO autonomous edits to `docs/index.html` / `README` / paper prose** (Public
   Documentation Discipline).
7. **Online ARC play stays operator-gated — NO leaderboard submission** in-loop. Only
   offline-reproduced levels count (`arc_solver_kit.reproduce`).
8. **Cross-domain SELECTION scope is RETIRED (exp4314)** and **cross-game value
   TRANSFER is RETIRED (exp4342)** — do NOT re-propose either.

---

## 7. SOTA references (this milestone; all WebFetch/WebSearch-verified 2026-06-17)

- **arXiv:2604.06260** — S³: Stratified Scaling Search for Test-Time in Diffusion
  Language Models. *The .402 HEADLINE instrument* (verifier-guided denoising search
  with the leak-robust scorer). Re-confirmed via WebSearch 2026-06-17.
- **arXiv:2606.08501** — PAPO / "Back on Track: Aligning Rewards and States for
  Reasoning in Diffusion LLMs." *The headline skeptic-proofing diagnostic* (reward-state
  alignment; authentic-state vs remasking).
- **arXiv:2605.05138** — Executable World Models for ARC-AGI-3 (Rodionov et al.;
  gpt-5.5 15/25, RHAE 58.12%). *The E3 north-star SOTA* (deeper/multi-game sweep).
- **arXiv:2605.25931** (AERA explore-verify-plan) + **arXiv:2512.22336** (Agent2World
  adaptive world-model testing). *The E3 loop structure* (explore-before-plan +
  adaptive tests) that produced the .401 ar25/sc25 solves; reused in PHASE B.
- **arXiv:2602.11146** (DiNa-LRM) — the leak-robust scorer substrate (exp4337), reused.
- **arXiv:2606.13565** (A2D2) / **arXiv:2502.01384** (SEPO) — verifier-as-reward
  generator FINE-TUNING; OUT-OF-BAND (operator-owned), flagged not auto-run.
- **arXiv:2605.15256** (ReactiveGWM) — full interaction-world-model cross-game transfer;
  deferred (the cheaper 3-null signal already retired the value-transfer line).

Cross-references: `ops/north-star.md` (§0/§1/§5), `ops/known-issues.md` (P0
oracle-distinct frontier — now MET; 2026-06-17 E3 mandate), `ops/arc_solve_registry.yaml`,
`research-references.md` (the .401/.402 sweep), `docs/research-notes/
diffusiongemma-energy-guided-diffusion-spec.md` (THE GATE — now MET).
