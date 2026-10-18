# Research Roadmap v248: Execution Sprint — Real-Corpus Verifier Recovery + Safety Classifier + KV260 Terminal + FR-11 Online

**Milestone:** 2026.05.248
**Previous Milestone:** 2026.05.247
**Date:** 2026-05-20
**Status:** PROPOSED
**ID allocation:** Milestone .247 used exp2569–exp2581; .248 starts at **exp2582**.

---

## What Milestone .247 Proved

Milestone .247 targeted three gaps from .246: real-corpus verifier recovery (tier0s/tier0u),
publication distribution mirroring (HuggingFace model cards + IPFS), and Tier B Safety Classifier
product track.

**Expected wins from .247:**

The .247 milestone was fully planned (13 tasks exp2569–exp2581, complete YAML, complete prompts)
but **completed 0 substantive experiments** (n_experiments_completed=0). Only the capstone
(exp2580) and retro (exp2581) ran, because they are the only tasks the outer-loop Claude agent
executes directly. The archive task (exp2569) also did not run.

This is the continuation of the **structural planning-execution gap**: 22+ consecutive milestones
in which substantive codex experiments were planned but not activated. The gap is a process
issue, not a research issue — the conductor requires a functioning codex backend to activate
tasks, and the backend has been in intermittent cascade failure.

**What .247 confirmed:**

1. **GateMate TERMINAL**: exp2580 capstone confirmed bitstream flashed, JTAG detects GM1Ax
   IDCODE 0x20000001. GateMate A1-EVB-2M has reached its terminal state per the Hardware-Task
   Continuity Discipline definition. Removed from mandatory per-milestone roster.
2. **AUROC unchanged**: headline AUROC 0.9857 (ensemble v7b, adversarially verified, 5-seed)
   remains best result — no regression, no improvement. Still beats HIVE peer (0.9236) and
   HalluScan peer mean (0.67).
3. **tier0s/tier0u still near-random**: real-corpus AUROC 0.3758 / 0.5360 unchanged. These
   two verifiers contribute near-zero information on natural text.
4. **KV260 still NOT TERMINAL**: no SD card, PYNQ URL unreachable. Blocked on operator action.
5. **Safety classifier still not implemented**: safety_classifier_viable=False.
6. **JEPA online integration still not done**: checkpoint trained in exp2565 but not wired into
   VerifyRepairPipeline.
7. **IPFS mirror not implemented**: Rule 3 compliance incomplete.
8. **HF model card citation not updated**: Rule 3 compliance incomplete.

---

## Three Biggest Gaps Entering .248

| # | Gap | .248 Fix |
|---|-----|---------|
| 1 | **Structural planning-execution gap**: 22+ consecutive milestones with 0 substantive experiments completed. All .247 work is technically unstarted. | No new scope additions — .248 is a pure execution sprint carrying forward .247 scope plus archive. If the gap persists, the operator must diagnose the conductor backend independently. |
| 2 | **tier0s/tier0u near-random on real corpus** (0.3758/0.5360): ensemble v7b AUROC 0.9857 is correct but two verifiers waste calibration signal | exp2583 (tier0s logistic regression retrain on real FoVer pairs) + exp2584 (tier0u NLI-proxy fix) → exp2591 (ensemble v9) |
| 3 | **KV260 still blocked**: only mandatory hardware task remaining after GateMate graduation | exp2590 (flash if SD card available; else update prep script); remains mandatory per Hardware-Task Continuity Discipline |

---

## Architecture: .248 Phase Structure

```
Phase 0 (Admin):
  exp2582 — Archive .247 + activate .248

Phase 1 (Publication Distribution — Rule 3):
  exp2583 [NOTE: renumbered from .247's exp2570] — HuggingFace model card citation update
  exp2584 [NOTE: renumbered from .247's exp2571] — IPFS mirror: pin arXiv preprint + generate CID

NOTE: exp2583/exp2584 above are the publication distribution tasks;
      the verifier recovery tasks are exp2585/exp2586 to avoid confusion.
```

Wait — let me use clean sequential numbering without cross-references to .247 task IDs:

```
Phase 0 (Admin):
  exp2582 — Archive .247 + activate .248

Phase 1 (Publication Distribution — Rule 3):
  exp2585 — HuggingFace model card citation update (arXiv preprint reference)
  exp2586 — IPFS mirror: pin arXiv preprint + generate CID

Phase 2 (Real-Corpus Verifier Recovery):
  exp2583 — tier0s real-corpus retrain: logistic regression on FoVer real pairs (target real AUROC > 0.65)
  exp2584 — tier0u NLI-proxy fix: TF-IDF cosine self-consistency on real text (target > 0.60)

Phase 3 (Tier B Safety Classifier — Product Track):
  exp2587 — Safety corpus 200 pairs + Tier0xSafetyVerifier (Ising pattern energy)
  exp2588 — Safety ensemble Group F + paper §7 stub [gated on exp2587.safety_verifier_viable]

Phase 4 (Continuous Self-Learning — FR-11):
  exp2589 — JEPA v3 online integration: real-data checkpoint → VerifyRepairPipeline with session-level update

Phase 5 (Hardware — MANDATORY):
  exp2590 — KV260 hardware continuity (flash if operator inserted SD card; prep script update otherwise)

Phase 6 (Ensemble v9 + Paper Update):
  exp2591 — Ensemble v9: incorporate real-corpus-validated tier0s/tier0u replacements
  exp2592 — Paper update: ensemble v9 results + safety §7 + distribution mirrors

Phase 7 (Synthesis):
  exp2593 — Capstone v248 [claude+opus; NO HARD GATE]
  exp2594 — Retro v248 [codex]
```

**Dependency graph:**

```
exp2582 (archive)
  → exp2583 (tier0s retrain) → exp2591 (ensemble v9)
  → exp2584 (tier0u fix) → exp2591 (ensemble v9)
  → exp2585 (HF cards) [independent]
  → exp2586 (IPFS) [independent]
  → exp2587 (safety corpus+verifier) → exp2588 (safety ensemble) [gated on exp2587.safety_verifier_viable]
  → exp2589 (JEPA v3 online) [independent]
  → exp2590 (KV260) [independent]
  → exp2591 (ensemble v9) [gated on exp2583.tier0s_real_auroc > 0.3758 OR exp2584.tier0u_real_auroc > 0.5360]
  → exp2592 (paper update) [reads exp2591 + exp2588]
  → exp2593 (capstone) [reads all]
  → exp2594 (retro)
```

**Critical path:** exp2583 + exp2584 → exp2591 → exp2593.

---

## Hardware Requirements

| Board | State Entering .248 | .248 Target |
|-------|--------------------|-----------:|
| AMD/Xilinx KV260 | SD card absent; PYNQ URL unreachable; operator docs produced in exp2560 | Flash if operator has inserted SD card; else update prep script with latest PYNQ URLs |
| Cologne Chip GateMate A1-EVB-2M | **TERMINAL**: bitstream flashed, JTAG confirms GM1Ax IDCODE 0x20000001 | Graduated — no longer mandatory per-milestone. On-board Ising sampler timing benchmark is OPTIONAL/opportunistic. |
| Microchip PolarFire SoC | **TERMINAL** (exp2501) | Optional/opportunistic only |

---

## Decentralization Compliance (Rules 1–7)

| Rule | Compliance |
|------|-----------|
| 1. Local-first open models | All training uses FoVer real corpus (local); no closed-weight calls |
| 2. Closed models optional | No closed-weight dependencies in exp2583–exp2591 |
| 3. Distribution mirroring | exp2585 (HF citation) + exp2586 (IPFS pin) directly implement Rule 3 for arXiv preprint |
| 4. Multiple integration surfaces | Safety verifier (exp2587) exposed via same API/CLI/MCP surfaces |
| 5. Hardware portability | KV260 track advances sovereignty; IPFS adds decentralized distribution |
| 6. Data minimization | No external LLM calls in scope; all data from local FoVer corpus or public safety benchmarks |
| 7. No vendor abstractions in core | All new code in `carnot/verify/` or `carnot/pipeline/` via abstract protocols |

---

## Exclusion Manifest Cross-Check

Retired experiment IDs in `ops/exclusion_manifest.yaml`: exp2091, exp260, exp308, exp309,
exp346, exp380–383, exp410, exp425, exp491, exp527, exp603, exp627, plus scope-retired
HalluSAEGeometricProbe, exp887/783/799/804/809/825 (discriminative JEPA OOD).

None of the .248 task scopes match any retired experiment ID or scope by name or deliverable
shape. No retired experiment IDs appear in any task's `requires:` chain.

---

## Failed-Experiment Rerun Compliance

Note: experiments exp2569–exp2579 (the .247 substantive tasks) were PLANNED but NEVER ACTIVATED
— they have no verdict in the results directory and are NOT on the exclusion manifest. They are
not "failed experiments" in the sense of the Failed-Experiment Rerun Discipline; they simply
never ran. The prior_failures entries below reference the most recent EXECUTED experiments
with matching scope.

| New Task | Closest Prior Executed | Scope Match | How Addressed |
|----------|-------------|-------------|---------------|
| exp2583 (tier0s retrain) | exp2509 (HalluGuard NTK, synthetic, AUROC=1.0 synth → real=0.3758) | PARTIAL: same verifier class | Real-corpus logistic regression on FoVer labeled pairs vs exp2509's synthetic NTK-proxy training. Training data source (real vs synthetic) is the structural distinction. retire_if_same_verdict=false (different gate metric: real AUROC > 0.3758 baseline) |
| exp2584 (tier0u fix) | exp2535 (tier0u logical consistency, synthetic AUROC=0.96 → real=0.5360) | PARTIAL: same verifier class | TF-IDF cosine self-consistency on real natural text vs exp2535's synthetic self-consistency. Training source AND scoring technique both changed. retire_if_same_verdict=false |
| exp2585 (HF model cards) | exp2570 (.247, never activated) | SCOPE MATCH: identical | exp2570 never ran — no verdict, not on exclusion manifest. This is the first execution attempt. retire_if_same_verdict=false |
| exp2586 (IPFS mirror) | exp2571 (.247, never activated) | SCOPE MATCH: identical | exp2571 never ran — no verdict, not on exclusion manifest. This is the first execution attempt. retire_if_same_verdict=false |
| exp2587 (safety corpus+verifier) | exp2574 (.247, never activated) | SCOPE MATCH: identical | exp2574 never ran. Also: exp2522 (HalluGuard corpus, hallucination domain) is the closest prior executed experiment. Different domain (safety/jailbreak vs hallucination), different labels (safe/unsafe vs hallucination/correct). retire_if_same_verdict=false |
| exp2589 (JEPA v3 online) | exp2565 (JEPA real training, .246) | PARTIAL: JEPA predictor | exp2565 trained the predictor; this integrates the checkpoint into VerifyRepairPipeline with online-update. Different deliverable: integration vs training. retire_if_same_verdict=false |
| exp2590 (KV260) | exp2578 (.247, never activated) | SCOPE MATCH: identical | exp2578 never ran. Also: exp2560 (KV260 operator docs, .246) is the closest prior executed experiment. exp2560 produced docs; this activates flash if SD available or updates prep scripts. Different deliverable: flash/prep vs docs. retire_if_same_verdict=false |
| exp2591 (ensemble v9) | exp2563 (ensemble v8, .246) + exp2579 (.247, never activated) | PARTIAL: ensemble expansion | exp2563 added Group E (tier0t/0v/0w new verifiers); exp2579 never ran. exp2591 replaces weak tier0s/tier0u with real-corpus-validated alternatives. Different: substitution vs addition. retire_if_same_verdict=false |

---

## New Literature Integration

The Post-.247 Planning Sweep (2026-05-20) identified two new relevant papers:

- **arXiv:2605.07209** (Hallucination Detection via Activations of Open-Weight Proxy Analyzers):
  Stacking ensemble over 7 analyzer architectures on 72k samples. The activation-feature approach
  is a candidate orthogonal signal for tier0s/tier0u repair — proxy analyzer activations may be
  more stable on real FoVer corpus than pure NTK/self-consistency approaches.

- **arXiv:2605.00323** (OSCAR — Online Self-Calibration Against Hallucination): Online MCTS
  with dual-granularity rewards for continual calibration without retraining. Strong comparator
  for exp2589 JEPA online integration; dual-granularity reward signal could inform JEPA's
  online_update() loss design.

Both papers added to `research-references.md` under "2026-05-20 Post-.247 Planning Sweep."

---

## Agent Routing

| Agent | Count | Fraction | Tasks |
|-------|-------|---------|-------|
| codex / gpt-5.5 | 12 | 92.3% | exp2582–2592, exp2594 |
| claude + opus | 1 | 7.7% | exp2593 (capstone) |

**requires_claude: true justification:**
- exp2593 (capstone): Multi-artifact synthesis across 12 experiment files + cross-phase judgment
  on real-corpus AUROC improvement, publication distribution status, safety classifier viability,
  KV260 hardware terminal state, FR-11 online integration status. Open-ended operator
  recommendation. Meets all 3 positive criteria.
