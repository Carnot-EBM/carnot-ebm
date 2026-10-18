# Research Roadmap v250: sklearn Fix + Verifier Recovery + Semantic Energy Tier 0z + Safety Tier B

**Milestone:** 2026.05.250
**Previous milestone:** 2026.05.249
**Date:** 2026-05-20
**Status:** PROPOSED

---

## What Milestone .249 Proved (5 wins, 3 gaps)

### Wins
1. **JEPA online integration complete** (exp2602): `online_update()` + `get_session_stats()` wired into `VerifyRepairPipeline`; `partial_fit` tested with synthetic observations. FR-11 Tier 3 pipeline integration mandate satisfied.
2. **Archive + activation clean** (exp2595): `.249` correctly identified as already-activated; milestone lifecycle mechanics working.
3. **Root cause isolated for verifier retrain failures** (exp2596/exp2597): both emitted `honest_verdict: blocked_sklearn` — confirming sklearn not installed is the single blocking dependency for 5+ planned tasks.
4. **Headline AUROC stable**: 0.9857 (ensemble v7b, adversarially verified) — no regressions.
5. **Research sweep yielded 5 new papers** — arXiv:2508.14496 (Semantic Energy Tier 0z), arXiv:2604.01473 (SelfGrader safety), arXiv:2601.03600 (ALERT jailbreak), arXiv:2603.23854 (Symbolic-KAN), arXiv:2505.19475 (Verifier-Driven TTT).

### Gaps
1. **sklearn not installed** — blocks tier0s retrain (exp2596 blocked), tier0u fix (exp2597 blocked), safety corpus build (exp2600 blocked), ensemble v9 (exp2604 gated). PRIMARY STRUCTURAL BLOCKER.
2. **Publication distribution unexecuted** — HF model card citation (exp2598) and IPFS mirror (exp2599) may not have run; Rule 3 compliance incomplete.
3. **25th consecutive empty-timing retro** — structural planning/execution gap persists; n_experiments_completed=0 in retro; root cause is not milestone planning but conductor activation.

---

## Three Biggest Gaps Between Current State and PRD Vision

### Gap 1: sklearn-blocked verifier recovery (HIGHEST PRIORITY)
**State:** tier0s real-corpus AUROC = 0.3758, tier0u = 0.5360 (both near-random). Three consecutive milestone attempts all emitted `blocked_sklearn`. The fix is trivial: `pip install scikit-learn` in the experiment environment. Once installed, both logistic-regression retrain tasks should run cleanly. Without this fix, ensemble v9 cannot be built and no verifier quality improvement is possible.

**Why it matters for PRD:** FR-12 (Verifiable Reasoning) requires that constraint satisfaction verification is reliable. Near-random verifiers degrade ensemble signal — two of 10 verifiers providing near-random scores dilute Fisher combination and inflate calibration error.

### Gap 2: Semantic Energy Tier 0z — no-training verifier for OOD robustness
**State:** The synthetic-to-real distribution collapse (tier0s 1.0→0.3758, tier0u 0.96→0.5360) demonstrates that trained-on-synthetic verifiers fail OOD. arXiv:2508.14496 (Semantic Energy) proposes a Boltzmann-inspired energy computed directly from logits — NO training required. This property makes it inherently OOD-robust (there is no training distribution to collapse on). The semantic clustering approach is implementable with TF-IDF cosine similarity and should generalize to natural FoVer text.

**Why it matters for PRD:** The "energy function as ground truth" invariant (CLAUDE.md Operational Principles) requires verifiers that don't game the gate by training on synthetic distributions. Training-free verifiers are the structural solution.

### Gap 3: Publication distribution trail incomplete (Rule 3)
**State:** arXiv package is ready (arxiv_ready_v4=True, exp2558). Operator must submit. But HF model cards haven't been updated with the citation, and IPFS mirror hasn't been pinned. Rule 3 (Distribution Mirroring) mandates HuggingFace as primary + IPFS as secondary. Neither has been executed as of .249.

**Why it matters for PRD:** Phase 1 ship gate includes "HuggingFace mirror per Rule 3 (mandatory mirroring)." The publication trail is load-bearing for the Phase 1 ship gate.

---

## Architecture Snapshot (entering .250)

```
Verifier Ensemble v7b (AUROC = 0.9857 on FoVer, adversarially verified):
  Group A (logprob): tier0a, tier0b, tier0c
  Group B (semantic): tier0d, tier0e, tier0f
  Group C (type/logic): tier0g, tier0h, tier0i
  Group D (Curry-Howard): tier0r (AUROC=0.9123)
  Group E (hallucination-specific): tier0t (dynamical), tier0v (HalluField proxy)

  WEAK verifiers (near-random on real corpus, pending retrain):
    tier0s (HalluGuard NTK proxy): real-corpus AUROC = 0.3758
    tier0u (Logical Consistency): real-corpus AUROC = 0.5360

  Pending .250:
    Tier 0z (Semantic Energy arXiv:2508.14496): training-free Boltzmann logit energy

FR-11 Self-Learning Stack:
  Tier 1: Online weight updates (CPU counter updates) — WIRED
  Tier 2: Constraint memory (pattern cache) — WIRED
  Tier 3: JEPA Predictor online_update() — WIRED (.249 exp2602, partial_fit tested)
  Tier 4: Adaptive energy (KAN structural) — PROTOTYPED

Hardware:
  GateMate A1-EVB-2M: TERMINAL (.247 capstone exp2580)
  KV260: NON-TERMINAL (SD card absent; synthesis succeeded; PYNQ path viable)
  PolarFire SoC: TERMINAL (.241 exp2501)
  RTX 3090 x2: available (0% GPU utilization in recent retros — no active GPU workloads)

Publication:
  arxiv_ready_v4 = True (exp2558, .246)
  arXiv ID: pending operator submission
  HF model cards: citation update NOT YET EXECUTED
  IPFS mirror: NOT YET EXECUTED
```

---

## Dependency Graph

```
exp2608 (archive .249 + activate .250)
    │
    ├── exp2609 (sklearn install + PYTHONPATH verify) ──────────┐
    │       │                                                    │
    │       ├── exp2610 (tier0s real retrain) ─────────────┐    │
    │       │       │                                       │    │
    │       ├── exp2611 (tier0u TF-IDF fix) ───────────┐   │    │
    │       │                                           │   │    │
    │       └── exp2613 (safety corpus + Tier0x) ─┐    │   │    │
    │                                               │    │   │    │
    ├── exp2612 (Tier 0z Semantic Energy) ──────────┤    │   │    │
    │                                               │    │   │    │
    ├── exp2614 (HF + IPFS distribution) ─(indep)  │    │   │    │
    │                                               │    │   │    │
    │             ┌──────────────────────────────────┘    │   │    │
    │             │                ┌────────────────────────┘   │    │
    │             ▼                ▼                             ▼    │
    │         exp2616 (safety Group F)               exp2615 (ensemble v9)
    │                                                    │
    ├── exp2617 (JEPA real-data evaluation) ─(indep)    │
    │                                                    │
    ├── exp2618 (KV260 continuity) ──────────(indep)    │
    │                                                    │
    └──────────────────────────────────────────────────► exp2619 (capstone)
                                                             │
                                                         exp2620 (retro)
```

---

## Phase Descriptions

### Phase 0: Archive and Activation (exp2608)
Archive milestone .249 into `research-complete.yaml`. Activate .250 by copying `research-roadmap-next.yaml` → `research-roadmap.yaml`.

### Phase 1: Prerequisite Fix (exp2609)
**sklearn install + PYTHONPATH verify** — the single root cause for 3+ blocked_sklearn experiments. Steps:
1. `pip install scikit-learn` in the active venv
2. Verify `python -c "import sklearn; print(sklearn.__version__)"` succeeds
3. Verify `python -c "import sys; sys.path.insert(0,'{project_root}/python'); import carnot"` succeeds
4. Record `sklearn_available: true` in artifact

This is the prerequisite gate for exp2610, exp2611, exp2613.

### Phase 2: Verifier Recovery (exp2610, exp2611, exp2612)
Three parallel verifier tracks:
- **exp2610** (tier0s real retrain): TF-IDF logistic regression on FoVer real corpus (n~6548 pairs). Target: AUROC > 0.65. Gate for ensemble v9 (exp2615).
- **exp2611** (tier0u NLI-proxy fix): TF-IDF cosine overlap as self-consistency proxy on real corpus. Target: AUROC > 0.60.
- **exp2612** (Tier 0z Semantic Energy): training-free Boltzmann energy from logits. Target: AUROC > 0.55. Per arXiv:2508.14496 — semantic cluster partition function approach.

### Phase 3: Safety Tier B + Distribution (exp2613, exp2614)
- **exp2613** (safety corpus + Tier0x): retry exp2600 with sklearn now available. 200 safe/unsafe pairs (HF wildguard or synthetic fallback). Shannon entropy + compliance_run_length features. Target: safety_auroc > 0.60.
- **exp2614** (HF + IPFS distribution): update `docs/model-card.md` with arXiv citation stub, pin `docs/arxiv-submission/` to IPFS (or generate SHA-256 placeholder), update README Distribution Channels section.

### Phase 4: Ensemble + Safety Integration (exp2615, exp2616)
- **exp2615** (ensemble v9): incorporate improved tier0s/tier0u pkl models + Tier 0z if viable. Target: AUROC ≥ 0.95 (allow distribution-shift drop from 0.9857).
- **exp2616** (safety Group F + §7): add Group F to ensemble.py with Fisher calibration; write paper §7 stub.

### Phase 5: Self-Learning + Hardware (exp2617, exp2618)
- **exp2617** (JEPA real-data eval): now that `online_update()` is wired (exp2602), evaluate it on real FoVer examples — run 50 examples through VerifyRepairPipeline with online learning active, measure `n_partial_fits`, `final_fast_path_rate`, `auroc_change`. Continuous self-learning mandate.
- **exp2618** (KV260 continuity): Branch A (SD detected → flash attempt), Branch B (SD absent → update prep script). Mandatory per Hardware-Task Continuity Discipline.

### Phase 6: Synthesis (exp2619, exp2620)
- **exp2619** (capstone, claude+opus): synthesize all .250 outcomes. Report: (a) sklearn fix resolved? (b) tier0s/tier0u improved? (c) Tier 0z viable? (d) ensemble v9 AUROC? (e) safety classifier viable? (f) distribution channels updated? (g) JEPA online learning measurable on real data? (h) operator recommendations.
- **exp2620** (retro, codex): operational retrospective, changelog/status update.

---

## Hardware Requirements

| Board | State | .250 Task | Next Blocker |
|---|---|---|---|
| KV260 | NON-TERMINAL | exp2618 (Branch A/B) | SD card absent |
| GateMate A1-EVB-2M | TERMINAL | None | Graduated |
| PolarFire SoC | TERMINAL | None | Graduated |
| RTX 3090 x2 | Available | exp2617 (JEPA eval) | None |

---

## Decentralization Compliance Check (CLAUDE.md Rules 1–7)

1. **Local-first open models** — experiments use FoVer corpus + logistic regression, no closed-weight dependencies. Tier 0z is training-free. ✓
2. **Closed-weight integration optional** — no closed-weight calls in any .250 task. ✓
3. **Distribution mirroring** — exp2614 implements HF + IPFS per Rule 3. ✓
4. **Multiple integration surfaces** — pipeline exposed via Python API + CLI + MCP server. ✓
5. **Hardware portability** — sklearn logistic regression runs on any CPU; no GPU required for .250. ✓
6. **Data minimization** — no closed-weight LLM calls. ✓
7. **No vendor abstractions in core** — all .250 code goes in `python/carnot/verify/`. ✓

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` against .250 task scopes:

| .250 Task | Pattern checked | Result |
|---|---|---|
| exp2610 tier0s retrain | HalluSAEGeometricProbe — different (that was SAE geometry) | CLEAR |
| exp2611 tier0u TF-IDF | discriminative JEPA OOD (783-887) — different (that was JEPA, this is logistic regression) | CLEAR |
| exp2612 Tier 0z | SpecAnn, THRML scaling sweep — different domains | CLEAR |
| exp2613 safety corpus | GRPO/VPRM, HardNet++ — different domains | CLEAR |
| exp2615 ensemble v9 | No matching retired ensemble patterns | CLEAR |
| exp2617 JEPA real eval | discriminative JEPA OOD (783-887) — exp2617 is online_update() evaluation (generation-path), NOT discriminative JEPA predictor training | CLEAR |
| exp2618 KV260 | iCE40 PIMI retired — different (that was PIMI optimization, this is SD flash/prep) | CLEAR |

Zero matches. Milestone activation not blocked by exclusion manifest.

---

## Failed-Experiment Rerun Compliance Table

| Task | Prior failure(s) | Addressed by |
|---|---|---|
| exp2610 (tier0s retrain) | exp2596 (blocked_sklearn) | sklearn installed via exp2609 prerequisite fix |
| exp2611 (tier0u fix) | exp2597 (blocked_sklearn) | sklearn installed via exp2609 |
| exp2612 (Tier 0z) | exp103 (early KAN energy tier — different arch) | Different arch (training-free Boltzmann, not KAN) |
| exp2613 (safety corpus) | exp2600 (blocked_sklearn) | sklearn installed via exp2609 |
| exp2614 (HF+IPFS) | exp2598/exp2599 (may not have run) | Consolidated into single task; adds explicit sklearn-check skip |
| exp2615 (ensemble v9) | exp2604 (gated, may not have run) | Gate satisfied when exp2610 succeeds |
| exp2616 (Group F) | exp2601 (gated on safety_verifier_viable) | Gate satisfied when exp2613 succeeds |
| exp2617 (JEPA eval) | exp2576/exp2589 (JEPA integration .247/.249) | Different scope: evaluation of online_update(), not re-integration |
| exp2618 (KV260) | exp2603 (planned .249, unknown result) | Same Branch A/B structure; new prompt field ensures conductibility |
| exp2619 (capstone) | exp2606 (capstone .249, 0 experiments) | .250 has sklearn fix — expect > 0 experiments |
| exp2620 (retro) | exp2607 (retro .249, 0 experiments) | Same |

---

## Agent Routing

| Task | Agent | Why |
|---|---|---|
| exp2608–exp2618 | codex + gpt-5.5 | Formulaic: install + retrain + eval + HF/IPFS + hardware. No multi-file synthesis. |
| exp2619 (capstone) | claude + opus | Multi-artifact cross-synthesis across 11 deliverables; open-ended operator recommendation under ambiguity. Meets all 3 positive-criterion conditions. |
| exp2620 (retro) | codex + gpt-5.5 | Templated operational retrospective. |

**Routing distribution:** 12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%) — codex-default discipline maintained.

---

## Critical Path

```
exp2609 (sklearn fix) → exp2610 (tier0s) + exp2611 (tier0u) + exp2613 (safety)
                      → exp2615 (ensemble v9) + exp2616 (Group F)
                      → exp2619 (capstone)
```

Fastest possible milestone closure requires exp2609 to run early and unblock the chain.

---

## What Success Looks Like for .250

- `sklearn_available: true` (exp2609) — unblocks the entire retrain chain
- `tier0s_real_auroc > 0.65` (exp2610) — real-corpus verifier recovery
- `tier0u_real_auroc > 0.60` (exp2611) — real-corpus verifier recovery
- `tier0z_auroc > 0.55` (exp2612) — training-free OOD verifier viable
- `safety_verifier_viable: true` (exp2613) — safety AUROC > 0.60
- `ensemble_v9_auroc >= 0.95` (exp2615) — ensemble not regressed after retrain
- `group_f_added: true` (exp2616) — safety ensemble integration
- `ipfs_cid` non-empty (exp2614) — Rule 3 compliance
- `jepa_partial_fits_on_real_data >= 1` (exp2617) — FR-11 validated on real corpus
- `n_experiments_completed >= 5` (exp2619) — break the 25-retro execution drought
