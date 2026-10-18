# Research Roadmap v395 — CLOSE the cross-game/cross-family OOD question: does the hardened ARC oracle-distinct selector win GENERALIZE beyond the pool it was tuned on?

**Milestone:** 2026.06.395
**Author:** `.395 planning sweep (Claude Opus 4.8, outer-loop), 2026-06-15
**Predecessor:** `openspec/change-proposals/research-roadmap-v394.md`
**North star:** `ops/north-star.md` §0 (solve ARC-AGI-3, accurately + efficiently) · §1 (FoVer headline) · §5 (energy VERIFIES, refinement GENERATES)

> **Architecture-freshness flag:** `_bmad/architecture.md` "Last Reconciled" is 2026-05-16 (exactly 30 days
> stale as of 2026-06-15). This milestone is DEPTH on an existing headline (no new capability / no architecture
> evolution), so it proceeds; the operator should schedule an architecture reconcile before the next *new
> capability* milestone.

---

## 0. THE HEADLINE QUESTION

**Does the hardened within-pool ARC oracle-distinct selector win (set_encoder@1 beats vote@1 by +44pp) survive
the REAL out-of-distribution test — held-out TASK-FAMILIES / GAMES it was never trained on?**

This is the ONE question `.394 could not answer. `.394 HARDENED the win on 2 of its 3 axes (it survived the
provenance-blind leak audit AND replicated across ≥5 seeds + an independent re-score), but the decisive
cross-game OOD test **BLOCKED** on a data-provenance bug (`blocked_arc_game_ids_unrecoverable`, exp4258): the
52-task grown pool's `task_id`s (`gap3_stage2:…`) could not be mapped back to game/family identities, so no
family-disjoint split could be built. The capstone therefore set `hardened_win=False` and held the
`diffusiongemma_full_run_gate` CLOSED — **not because the win collapsed, but because the load-bearing OOD test
never ran.** Closing it is decision-grade either way:

- **GENERALIZES** (cross-family delta>0, CI95-excl-0) → the selector is a GENERAL signal, north-star-grade
  (the verifier earns its place in the ARC-AGI-3 harness, which faces new games/levels constantly), the win
  becomes headline-eligible, and the DiffusionGemma scale-up gate can open.
- **COLLAPSES** (delta≈0 cross-family) → the +44pp is per-family-basin memorization; we SCOPE the headline to
  "within-pool selection" and refocus on a generalizing verifier. (Carnot's own prior evidence warns this is
  where learned ARC signals collapse: GAP-4 ARC-1→ARC-2 induction 0.93→0.57, `project_gap3_verifier_program`.)

`.395 is DEPTH, not churn (north-star §1, Depth-Over-Breadth ethos). It does ONE thing — close the cross-family
question — and stages everything else around that answer.

---

## 1. WHAT `.394 PROVED (and the ONE gap `.395 must close)

| Axis | `.394 result | Status for `.395 |
|---|---|---|
| Leak-audit (A1, exp4256) | Win SURVIVES provenance-blind features: de-leaked delta **+0.385**, AUROC **0.990** (origin_probe_auroc 0.948 — origin WAS encoded, but stripping it still leaves a real content-signal win) | ✅ HARDENED — not re-run |
| Multi-seed (A2, exp4257) | REPLICATES: mean delta **+0.458** across ≥5 seeds; independent re-score **+0.442** matches single-seed exactly; oracle@K 0.827, n=52 | ✅ HARDENED — not re-run |
| **Cross-game OOD (A3, exp4258)** | **BLOCKED** `blocked_arc_game_ids_unrecoverable` — the real OOD test NEVER RAN | ❌ **THE `.395 DEPTH TARGET** |
| Synthesis (A4, exp4259) | Does NOT break the oracle@K ceiling: synthesis_minus_oracle **−0.283** (beats vote +0.348 but underperforms selection). Selection IS the ceiling on this corpus | ⏸ Deprioritized — per-cell reconciliation is a dead end here; breaking oracle@K needs better candidate GENERATION, not reconciliation |
| DiffusionGemma preflight (B1, exp4260) | **BLOCKED** `blocked_diffusiongemma_gguf_loader_failed` (flagged CRITICAL, 0.34s) — the discrete-diffusion GGUF loader never worked | 🔧 `.395 INFRA FIX (loader repair + re-preflight) so the scale-up is READY the moment the win generalizes |
| ARC north star (B2/B3, exp4261/4262) | No advance: 19 levels held (no verifier-validated level-up on sc25 L6); live env 0 levels (efficiency-only) | ➕ `.395 targets +1 on a NEW game |
| Verifier-as-reward (C1, exp4263) | RE-SCOPED OUT-OF-BAND: `ready_for_out_of_band=true` — corpus+runner+validation prepared for the operator (TRM-checkpoint pattern) | ✅ OPERATOR-OWNED — NOT a loop task; recorded as retired-from-loop |
| Code oracle-distinct (C2, exp4264) | CORPUS-SPECIFIC: fresh-corpus predictor delta **−0.006** (did NOT replicate), off-fold AUROC 0.697 | ⛔ RETIRE the code-replication ask — the .392 +3.1pp stands single-corpus; no 3rd attempt |
| Publication gate | `paper_ready=True` (G1∧G2∧G3∧G4; FoVer 0.9131 headline) | ✅ Unchanged — the publication target |

**SOTA already ingested for `.395 (exp4265, real arXiv IDs, verified 2026-06-15):**
- **ARC-TGI — Human-Validated Task Generators (arXiv:2603.05099).** 461 task-family generators preserving a
  latent rule, covering 180 ARC-Mini + 215 ARC-AGI-1 tasks, each with family IDs + reasoning-chain templates.
  THE tool for a family-disjoint cross-game test → Phase A. (`flagged_for_v395` #1.)
- **Reliability Gap — benchmark-auditing provenance discipline (arXiv:2606.03305).** Contamination detectors
  fail under shift/small-n; *transparent provenance* (source-kind + family + fold + target-hash as first-class
  columns) must be the acceptance gate, not a post-hoc statistical detector. Directly motivated by exp4256's
  high origin_probe_auroc (0.948) → Phase A provenance manifest.
- **DPRM token-ordering guidance for diffusion LMs (arXiv:2604.24357)** + **Entropy-guided step selection for
  diffusion-LLM RL (arXiv:2603.12554)** → Phase B DiffusionGemma guided-reveal smoke (after loader repair).
- **L-VARC language-guided abstraction, inference-time visual backbone (arXiv:2606.12847)** → Phase A
  ablation: train with privileged family/template features, discard them at inference, score held-out-family
  exact match (a leak-control on the cross-family test).

---

## 2. THE PLAN — 4 phases + archive (11 tasks, exp4269–exp4279)

### PHASE A — CLOSE THE CROSS-FAMILY / CROSS-GAME OOD QUESTION (the depth target)

- **exp4270 (A1) — FAMILY/GAME PROVENANCE RECOVERY + diversity audit.** Fix exp4258's root cause: recover a
  task-FAMILY id for every row in the grown pool (join `raw_task_id`/`source_id` → the source ARC task → the
  ARC-TGI family taxonomy / the 25-game survey). Materialize a first-class provenance manifest
  (source-kind + family_id + game_id + fold + target-hash) per the Reliability Gap discipline. Emit
  `family_split_feasible` BARE bool := (≥4 distinct families, each held-out fold ≥~10 tasks for bootstrap
  power) + `distinct_family_n`. CPU-only. *Decision-grade on its own: feasible → A2 tests the existing pool;
  infeasible → A3 builds a fresh ARC-TGI family-diverse pool.*
- **exp4271 (A2) — CROSS-FAMILY TRANSFER on the EXISTING pool.** `gated_on` A1 `family_split_feasible==true`.
  Train the set-encoder on a subset of families, test the beats-vote gate on candidates from HELD-OUT
  families. Report `cross_family_delta` + bootstrap CI95 + `held_out_family_n` + `within_minus_cross_gap` +
  oracle@K positive control. Includes a STATIC vs ONLINE-ADAPTED arm (the self-learning probe). Emit
  `cross_family_win_holds` BARE bool. `verifier_is_oracle=false`.
- **exp4272 (A3) — FRESH ARC-TGI FAMILY-DISJOINT POOL + cross-family transfer.** `gated_on` A1
  `family_split_feasible==false` (the fallback when the 52-task pool is too family-concentrated). Build a
  larger family-diverse candidate pool from cached sources / ARC-TGI generators with materialized family IDs
  + exact target hashes; run the SAME family-disjoint gate. Keeps original ARC tasks as a held-out sanity
  read (per the ARC-TGI failure-mode warning: generator data can become its own distribution). Emit
  `cross_family_win_holds` BARE bool. `verifier_is_oracle=false`.
- **exp4273 (A4) — SELF-LEARNING (Tier-1 online verifier-weight adaptation) across the family folds.**
  `gated_on` A1 `family_split_feasible==true`. The mandated continuous-self-learning experiment
  (research-program.md Tier 1, CPU counter/weight updates — NOT the retired live-LoRA fine-tuning). On
  held-out families, compare the STATIC set-encoder selector vs an ONLINE-reweighted selector (upweight
  features/verifiers that selected correctly on the nearest seen family). Does cheap online adaptation
  recover cross-family headroom the static selector loses? `verifier_is_oracle=false`.

### PHASE B — REPAIR THE DiffusionGemma LOADER + RE-PREFLIGHT (infra de-risk; NO full run)

- **exp4274 (B1) — DiffusionGemma discrete-diffusion GGUF LOADER FIX + re-preflight.** Fix exp4260's root
  cause (`gguf_loader_failed`): get DiffusionGemma loading via the `.gguf` path (NOT AutoTokenizer on a GGUF
  repo). Re-run the TINY guidance smoke: wire the verifier ensemble as a per-step guidance energy, confirm it
  reweights token selection vs unguided (DPRM 2604.24357 / entropy-step 2603.12554), emit `preflight_go` +
  `guidance_changes_selection` + `full_run_cost_estimate_s`. **NO full benchmark, NO training** — the full
  run stays DEFERRED to `.396, gated on `hardened_win` (operator twice-burned by premature scale-up).
  `verifier_is_oracle=false`.

### PHASE C — ARC NORTH STAR (accuracy progress)

- **exp4275 (C1) — offline ARC incremental +1 on a NEW game.** Fix exp4261's wall (no level-up candidate on
  sc25 L6) by targeting the best-headroom UNATTEMPTED game from the survey, with the hardened set-encoder
  routing the solver's candidate ranking. Monotonic: `total_levels ≥ 20`, `levels_completed ≥ 1`
  real-env-confirmed. Per ARC-AGI-3 Incremental-Progress Scoping (+1, not all-levels). NO TRM training; NO
  leaderboard submission. (The live-env probe is intentionally NOT re-run standalone — it has measured
  0-levels-efficiency-only 4× and is churn until the offline solver advances; it resumes in `.396 only if C1
  advances offline.)

### PHASE D — HYGIENE, DECISIONS & CAPSTONE

- **exp4276 (D1) — SOTA-ingestion → `.396 forks.** Reliable channel only (`/deep-research` BANNED in-loop).
  Ingest SOTA for whichever way the cross-family question resolves; map the strongest 3–5 methods (real arXiv
  IDs) to `.396; emit `flagged_for_v396`.
- **exp4277 (D2) — verifier registry/gaps hygiene + GAP-4 regression guard + RECORD the `.394/`.395
  retirements** (code oracle-distinct → corpus-specific/retired; verifier-as-reward → out-of-band
  operator-owned). Log new missing-verifier gaps (e.g. a cross-family ARC selection gap if A2/A3 collapse).
  Emit `regression_guard_passed` BARE bool.
- **exp4278 (D3) — hardware continuity (opportunistic, north-star §3).** KV260 SSH-only + `xmutil listapps`
  (NEVER a host SD-card precondition), PolarFire CPU-dispatch hash-verify, GateMate USB detect. No board
  blocks the milestone.
- **exp4279 (D4) — CAPSTONE `.395 (UNGATED).** Headline question: did the win GENERALIZE cross-family? Emit
  `cross_family_generalizes` + `hardened_win` (now requires cross-family delta>0) + `diffusiongemma_full_run_gate`
  (resolvable iff `hardened_win` AND `preflight_go`). SKIP any `flagged_adversarial` artifact; HONOR
  `verifier_is_oracle`; compute G1–G4 via `publication_gate.py`.

---

## 3. DEPENDENCY GRAPH

```
exp4269 archive ─────────────────────────────────────────────────────────────► (enables all)

PHASE A (depth):
  exp4270 A1 provenance-recovery ──► family_split_feasible? ──┬─[true]─► exp4271 A2 cross-family (existing pool)
                                                              │         └─► exp4273 A4 self-learning online-adapt
                                                              └─[false]► exp4272 A3 fresh ARC-TGI pool + cross-family
        (exactly one of A2 / A3 produces cross_family_win_holds)

PHASE B (infra, ungated):   exp4274 B1 DiffusionGemma loader-fix + preflight ──► preflight_go (full run → .396)
PHASE C (north star):       exp4275 C1 ARC +1 (new game) ──► total_levels ≥ 20
PHASE D:                    exp4276 D1 SOTA → .396 │ exp4277 D2 registry/gaps │ exp4278 D3 hardware
CAPSTONE (ungated):         exp4279 D4 reads A2|A3 cross_family_win_holds + B1 preflight_go ──►
                                       cross_family_generalizes + hardened_win + diffusiongemma_full_run_gate
```

Gates (BARE-value, conjunctive): A2 ⟸ A1.family_split_feasible==true · A3 ⟸ A1.family_split_feasible==false ·
A4 ⟸ A1.family_split_feasible==true. B1/C1/capstone ungated.

---

## 4. HARDWARE REQUIREMENTS

- **Phase A (exp4270–4273):** CPU-only — cached grown pool + set-encoder retraining (exp4257 did 5 seeds in
  ~295s). No GPU, no GGUF.
- **Phase B (exp4274):** the cached DiffusionGemma GGUF (`unsloth/diffusiongemma-26B-A4B-it-GGUF`, confirmed
  present 2026-06-15); a TINY vocab/smoke load — minutes, not a full run. RTX 3090 available; preflight is
  load + a few denoising steps.
- **Phase C (exp4275):** CPU offline solver + cached ARC fixtures. NO TRM training (conductor stood down on
  TRM; val 0.8227 checkpoint is DONE).
- **Phase D:** aggregation (CPU) + SSH/USB hardware probes (KV260 `ssh kria`, PolarFire `ssh polarfire`,
  GateMate `openFPGALoader --detect`).

---

## 5. DISCIPLINES APPLIED (every task)

- **Circularity / Oracle-Distinctness:** every verifier-value task declares `verifier_is_oracle` honestly. The
  cross-family selector is `verifier_is_oracle=false` (learned, no demo execution). A leaked/circular result
  may NOT headline a moat (`adversarial_verify.py:check_circular_moat_overclaim`).
- **Adversarial Artifact Verification + Sample-Size Rigor:** the +44pp win is the program's #1 positive;
  cross-family is the surprising-result cross-check. Bootstrap CI95 ≥2000 resamples; positive control
  (oracle@K vs vote) to avoid FALSE_NEGATIVE_RISK on a no-headroom null. Capstone SKIPS `flagged_adversarial`.
- **Provenance discipline (Reliability Gap 2606.03305):** A1 materializes a first-class provenance manifest;
  every cross-family claim must trace rows to a source-kind + family + fold (not a post-hoc leak detector).
- **Failed-Experiment Rerun + Pre-Launch Preconditions:** cross-family tasks carry `prior_failures` on exp4258
  (root cause = game_ids_unrecoverable; fix = provenance recovery / ARC-TGI fresh pool; `retire_if_same_verdict`).
  B1 carries `prior_failures` on exp4260 (loader fix). C1 on exp4261 (new game + hardened routing). Every
  compute-bound task has a PRECONDITIONS step-0 with `blocked_<resource>` fallback.
- **Codex-Default v2 / Inference-Substrate / Principle-Annotated fields / Verdict Terminal-Prefix / gated-fields-bare**
  on every task. Online ARC play stays operator-gated (NO leaderboard submission). The publication target
  stays the FoVer headline (paper_ready=True); the ARC cross-family result is a NEW supporting/headline result.
