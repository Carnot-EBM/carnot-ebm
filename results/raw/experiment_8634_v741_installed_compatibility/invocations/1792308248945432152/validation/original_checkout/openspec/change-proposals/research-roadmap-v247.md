# Research Roadmap v247: Real-Corpus Verifier Recovery + Publication Distribution + Safety Classifier Tier B

**Milestone:** 2026.05.247
**Previous Milestone:** 2026.05.246
**Date:** 2026-05-20
**Status:** PROPOSED
**ID allocation:** Milestone .246 used exp2556–exp2568; .247 starts at **exp2569**.

---

## What Milestone .246 Proved

Milestone .246 targeted three gaps from .245: paper errata (tier0s/tier0u inflated synthetic AUROCs),
hardware terminal states (GateMate strtol fix + KV260 docs), and new verifier expansion (tier0t/0v/0w
+ ensemble v8).

**Expected wins from .246:**

1. **Paper errata applied** (exp2557): tier0s corrected from 1.0 → 0.3758 real-corpus, tier0u from
   0.96 → 0.5360, with synthetic-only labeling. Headline AUROC 0.9857 (ensemble v7b) unchanged.
2. **arXiv Final Package v4** (exp2558): arxiv_ready_v4=True with errata incorporated. Operator
   submission checklist produced; the operator can now submit to arXiv cs.AI / cs.LG.
3. **PYTHONPATH fix universally applied**: exp2561 (tier0t) and exp2562 (tier0v + tier0w) both used
   sys.path.insert(0, project_root/python) — root cause of .245 Tier0v failure resolved.
4. **JEPA real FoVer training** (exp2565, continuous_self_learning_task): JEPAFastPathPredictor
   trained on n=6548 real examples; checkpoint saved; AUC target >0.889 (synthetic baseline).
5. **HalluScan external benchmark** (exp2566): Carnot ensemble v7b evaluated on HalluScan domains;
   peer comparison vs published NLI baseline (mean AUROC 0.67) established.

**What .246 likely did NOT fully accomplish:**

- **GateMate flash uncertain**: exp2559 (CC1 toolchain or openFPGALoader HEAD) may or may not have
  resolved the strtol parse error. If approach A (CC1 toolchain) + approach B (HEAD build) both
  failed, the board is still in flash-pending state.
- **KV260 flash still operator-blocked**: exp2560 produced operator docs with confirmed-reachable PYNQ
  URLs, but physical SD card insertion + flash requires operator action between sessions.
- **tier0s/tier0u remain near-random on real corpus**: The paper errata corrects the claims but does
  NOT fix the underlying verifiers. Real-corpus AUROC remains 0.3758 / 0.5360 — these verifiers
  contribute near-zero information to the ensemble on natural text.

---

## Three Biggest Gaps Entering .247

| # | Gap | .247 Fix |
|---|-----|---------|
| 1 | tier0s/tier0u near-random on FoVer real corpus (0.38/0.54) — ensemble v7b AUROC 0.9857 is correct but these two verifiers waste calibration signal | exp2572 (diagnose root cause) + exp2573 (tier0s retrain on real FoVer) + exp2574 (tier0u real-text fix) → exp2579 (ensemble v9 with improved verifiers) |
| 2 | Publication distribution trail: after arXiv submission, HuggingFace model cards need paper citation + IPFS mirror (Rule 3) | exp2570 (HF model cards), exp2571 (IPFS pin arXiv preprint + generate CID) |
| 3 | Tier B Safety/Jailbreak Classifier: next commercially viable product; energy infrastructure exists; needs corpus + verifier implementation | exp2574 (safety corpus), exp2575 (safety energy verifier + ensemble integration), paper §7 stub |

---

## Architecture: .247 Phase Structure

```
Phase 0 (Admin):
  exp2569 — Archive .246 + activate .247

Phase 1 (Publication Distribution — Rule 3):
  exp2570 — HuggingFace model card citation update (arXiv preprint reference)
  exp2571 — IPFS mirror: pin arXiv preprint + generate CID

Phase 2 (Real-Corpus Verifier Recovery):
  exp2572 — Tier 0s real-data retraining: NTK-proxy on FoVer real pairs (target real AUROC > 0.65)
  exp2573 — Tier 0u real-data fix: improved self-consistency scoring on real text (target > 0.60)

Phase 3 (Tier B Safety Classifier — Product Track):
  exp2574 — Safety corpus + Ising safety verifier (200 safe/unsafe pairs; Ising energy)
  exp2575 — Safety ensemble integration + paper §7 stub [gated on exp2574.safety_verifier_viable]

Phase 4 (Continuous Self-Learning):
  exp2576 — JEPA v3 online integration: real-data checkpoint → VerifyRepairPipeline with session-level update

Phase 5 (Hardware — MANDATORY):
  exp2577 — GateMate hardware continuity (smoke test if terminal from .246; manual .cfg repair if not)
  exp2578 — KV260 hardware continuity (Carnot validation if operator flashed; prep script otherwise)

Phase 6 (Ensemble v9 + External Benchmark):
  exp2579 — Ensemble v9: incorporate real-corpus-validated tier0s/tier0u replacements; re-evaluate vs HalluScan

Phase 7 (Synthesis):
  exp2580 — Capstone v247 [claude+opus; NO HARD GATE]
  exp2581 — Retro v247 [codex]
```

**Dependency graph:**

```
exp2569 (archive)
  → exp2570 (HF cards) [independent]
  → exp2571 (IPFS) [independent]
  → exp2572 (tier0s retrain) → exp2579 (ensemble v9)
  → exp2573 (tier0u fix) → exp2579 (ensemble v9)
  → exp2574 (safety corpus+verifier) → exp2575 (safety ensemble) [gated on exp2574.safety_verifier_viable]
  → exp2576 (JEPA v3 online) [independent]
  → exp2577 (GateMate) [independent]
  → exp2578 (KV260) [independent]
  → exp2579 (ensemble v9) [gated on exp2572.tier0s_improved OR exp2573.tier0u_improved]
  → exp2580 (capstone) [reads all]
  → exp2581 (retro)
```

**Critical path:** exp2572 + exp2573 → exp2579 → exp2580.

---

## Hardware Requirements

| Board | State Entering .247 | .247 Target |
|-------|--------------------|-----------:|
| Cologne Chip GateMate A1-EVB-2M | Uncertain: exp2559 may have fixed strtol OR still flash-pending | Smoke test on-board Ising sampler if terminal; else manual .cfg token repair + reflash |
| Xilinx KV260 | SD media absent; operator docs produced in exp2560 | Carnot IsingVerifier validation on board if operator inserted SD + flashed; else prep-script update |
| Microchip PolarFire SoC | **TERMINAL** (exp2501) | Optional/opportunistic only |

---

## Decentralization Compliance (Rules 1–7)

| Rule | Compliance |
|------|-----------|
| 1. Local-first open models | All training uses FoVer real corpus (local); no closed-weight calls |
| 2. Closed models optional | No closed-weight dependencies in exp2572–2579 |
| 3. Distribution mirroring | exp2570 (HF citation) + exp2571 (IPFS pin) directly implement Rule 3 for arXiv preprint |
| 4. Multiple integration surfaces | Safety verifier (exp2574) exposed via same API/CLI/MCP surfaces |
| 5. Hardware portability | GateMate + KV260 tracks advance sovereignty; IPFS adds decentralized distribution |
| 6. Data minimization | No external LLM calls in scope; all data from local FoVer corpus or public safety benchmarks |
| 7. No vendor abstractions in core | All new code in `carnot/verify/` or `carnot/pipeline/` via abstract protocols |

---

## Exclusion Manifest Cross-Check

Retired experiment IDs in `ops/exclusion_manifest.yaml`: exp2091, exp260, exp308, exp309,
exp346, exp380–383, exp410, exp425, exp491, exp527, exp603, exp627, plus scope-retired
HalluSAEGeometricProbe, exp887/783/799/804/809/825 (discriminative JEPA OOD).

None of the .247 task scopes match any retired experiment ID or scope by name or deliverable
shape. No retired experiment IDs appear in any task's `requires:` chain.

---

## Failed-Experiment Rerun Compliance

| New Task | Closest Prior | Scope Match | How Addressed |
|----------|-------------|-------------|---------------|
| exp2572 (tier0s retrain) | exp2509 (HalluGuard NTK, synthetic training, AUROC=1.0 synth) | PARTIAL: same verifier class | Real-corpus training on FoVer labeled pairs vs exp2509's synthetic corpus. Training data source change is the structural distinction; retire_if_same_verdict=false (different gate metric: real-corpus AUROC not synthetic) |
| exp2573 (tier0u fix) | exp2535 (tier0u logical consistency, synthetic AUROC=0.96) | PARTIAL: same verifier class | Real-data training with NLI-proxy scoring vs exp2535's synthetic self-consistency. Training source and implementation technique both changed. retire_if_same_verdict=false |
| exp2574 (safety corpus) | exp2522 (HalluGuard corpus, hallucination domain) | PARTIAL: corpus construction | Different domain (safety/jailbreak vs hallucination). Different labels (safe/unsafe vs hallucination/correct). retire_if_same_verdict=false |
| exp2577 (GateMate) | exp2559 (CC1 toolchain / openFPGALoader HEAD, .246) | MATCH: same board | exp2559 tried CC1+HEAD; this tries manual .cfg token repair (inspect strtol token, edit, re-flash) if exp2559 failed, OR smoke test if exp2559 succeeded. retire_if_same_verdict=true if flash still fails |
| exp2578 (KV260) | exp2560 (KV260 operator docs, .246) | PARTIAL: same board | exp2560 produced docs; this tasks activates flash sequence if operator has inserted SD card, or updates prep scripts. retire_if_same_verdict=false (different deliverable: flash vs docs) |
| exp2579 (ensemble v9) | exp2563 (ensemble v8, .246) | MATCH: ensemble expansion | Structurally different: v8 added Group E (tier0t/0v/0w new verifiers); v9 replaces weak tier0s/tier0u with real-corpus-validated alternatives. Different verifier substitution vs addition. retire_if_same_verdict=false |
| exp2576 (JEPA v3 online) | exp2565 (JEPA real training, .246) | PARTIAL: JEPA predictor | exp2565 trained the predictor; this integrates the checkpoint into VerifyRepairPipeline with online-update capability. Different deliverable: integration vs training. retire_if_same_verdict=false |

---

## Agent Routing

| Agent | Count | Fraction | Tasks |
|-------|-------|---------|-------|
| codex / gpt-5.5 | 12 | 92.3% | exp2569–2579, 2581 |
| claude + opus | 1 | 7.7% | exp2580 (capstone) |

**requires_claude: true justification:**
- exp2580 (capstone): Multi-artifact synthesis across 12 experiment files + cross-phase judgment
  on real-corpus AUROC improvement, publication distribution status, safety classifier viability,
  hardware terminal states. Open-ended operator recommendation. Meets all 3 positive criteria.
