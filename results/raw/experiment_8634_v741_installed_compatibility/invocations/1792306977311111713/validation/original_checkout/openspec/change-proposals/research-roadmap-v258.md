# Research Roadmap — Milestone 2026.05.258
# Pre-Test Cascade Fix v1 + Phase 1 Ship v5 + GGUF Live Eval v3 + ODAR Routing + FR-11 ORCA TTT v2

**Date:** 2026-05-20  
**Prev Milestone:** 2026.05.257 (3 of 13 tasks completed: exp2699 archive, exp2700 conductor postmortem, exp2704 scaling audit)  
**This Milestone:** 2026.05.258  
**Experiment IDs:** exp2712–exp2724 (13 tasks)

---

## What Milestone .257 Proved

Three tasks landed artifacts in .257:

1. **exp2699** (Archive .256 + Activate .257): Completed. research-roadmap.yaml now shows 2026.05.257.
2. **exp2700** (Conductor Postmortem v2): ROOT CAUSE IDENTIFIED after 51 consecutive zero-execution milestones.
   - **PRIMARY cause:** `tests/python/inference/test_hw_dab.py` (introduced 2026-05-16 by exp2133, commit 8ade7c530) imports `torch` as a top-level statement. Torch is NOT installed in the `.venv`. When the conductor's smart subset pre-test picks up this file (any time `git diff --name-only HEAD~1` shows it), pytest collection fails → `run_tests()` returns False → every task SKIPs.
   - **AMPLIFIER:** `scripts/research_conductor.py:4085` sets `MAX_HEAL_ATTEMPTS = 0` (operator emergency 2026-05-03). With self-heal disabled, a single pre-test failure cascades into permanent SKIP for every task.
   - **Recovery commands:** copy-pasteable (pip install torch CPU, patch test file, clear pretest cache). Documented in `results/experiment_2700_conductor_postmortem_v2.json`.
3. **exp2704** (Multi-Agent Scaling Audit): `saturation_k=2`, `saturation_auroc=0.993`, `me_adequate=true`. This high AUROC at k=2 with negative total_lift suggests behavioral entanglement in the current verifier mix — motivates exp2723 (de-entangled reweighting).

**Ten tasks produced no artifacts** (exp2701–exp2703, exp2705–exp2711). All carry forward to .258.

---

## Three Biggest Gaps

### Gap 1: Conductor Pre-Test Cascade — Structural Fix Required (P0)
- **State:** Root cause known. Recovery commands in exp2700 artifact. Fix is mechanical.
- **Risk:** All research tasks continue to SKIP until the cascade is repaired.
- **Action (exp2713):** Install torch CPU wheel in `.venv`; patch `tests/python/inference/test_hw_dab.py` to use `pytest.importorskip('torch')` (safe long-term fix); clear `ops/.pretest-cache.json`.
- **Gate:** All Phase B–D experiments are `gated_on: exp2713.pretest_cascade_fixed == true`. If the fix fails, they skip cleanly rather than wasting Sonnet budget.

### Gap 2: GGUF Live Pipeline Validation — Both RTX 3090s Idle
- **State:** Both RTX 3090s (48 GB VRAM) have been idle since ~.206. No live inference validation has ever run.
- **Prior attempts:** exp2675 (.255), exp2689 (.256), exp2702 (.257) — all zero_execution due to conductor stall.
- **Action (exp2715):** With cascade fixed, run VerifyRepairPipeline on N=50 live GGUF outputs from Qwen3.6-35B or Gemma-4-31B. Full PRECONDITIONS block first.

### Gap 3: Phase 1 Ship — Still HOLD
- **State:** README.md Phase 1 section absent; RELEASES.md absent; no git tag.
- **Prior attempts:** exp2674 (.255), exp2688 (.256), exp2701 (.257) — all zero_execution.
- **Action (exp2714):** With cascade fixed, execute autonomous ship prep. README + RELEASES.md + operator_ship_checklist_v5 with copy-pasteable `git tag` command.

---

## Architecture Diagram (Current .258 State)

```
                      ┌─────────────────────────────────────────────────────┐
                      │           MILESTONE 2026.05.258                     │
                      │         exp2712 – exp2724                           │
                      └─────────────────────────────────────────────────────┘

PHASE A: Infrastructure + Activation (exp2712–exp2714)
┌──────────────┐   ┌─────────────────────────────┐   ┌────────────────────┐
│  exp2712     │   │  exp2713 [P0]               │   │  exp2714           │
│  Archive     │──▶│  Pre-Test Cascade Fix v1    │   │  Phase 1 Ship v5   │
│  .257 +      │   │  torch CPU install +        │   │  README + RELEASES │
│  Activate    │   │  test patch + cache clear   │──▶│  + checklist v5    │
│  .258        │   │  pretest_cascade_fixed: bool│   │  [gates on .fixed] │
└──────────────┘   └────────────────┬────────────┘   └────────────────────┘
                                    │ gated_on
                    ┌───────────────┴──────────────────────────────────────┐
                    ▼               ▼               ▼               ▼
PHASE B: Live Evaluation + Verifier Stack (exp2715–exp2718)
┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│  exp2715     │   │  exp2716      │   │  exp2717       │   │  exp2718     │
│  GGUF Live   │   │  Tier 0f      │   │  Property      │   │  Linear Probe│
│  Eval v3     │   │  Semantic     │   │  Guided        │   │  Calibration │
│  N=50        │   │  Calibration  │   │  Counterexample│   │  v2 (10x     │
│  RTX 3090    │   │  v2           │   │  Repair v2     │   │  speedup)    │
└──────────────┘   └───────────────┘   └────────────────┘   └──────────────┘
                    ▼               ▼               ▼               ▼
PHASE C: FR-11 + Theory + New Research (exp2719–exp2721)
┌──────────────┐   ┌───────────────┐   ┌────────────────────────────────────┐
│  exp2719     │   │  exp2720      │   │  exp2721                           │
│  FR-11 Tier 3│   │  ODAR Free-   │   │  Paper v6 Theory Update v2         │
│  ORCA TTT v2 │   │  Energy +     │   │  ARM-EBM bijection §2 + 4/δ §3    │
│  + Grounded  │   │  T2 VegAS     │   │  + FST §3 (precondition-gated on   │
│  Continuation│   │  K-Scaling    │   │  pdflatex)                         │
│  [FR-11 ✓]   │   │  [uses 2704]  │   │                                    │
└──────────────┘   └───────────────┘   └────────────────────────────────────┘

PHASE D: Hardware + New Research + Capstone (exp2722–exp2724)
┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
│  exp2722     │   │  exp2723                  │   │  exp2724             │
│  KV260       │   │  Behavioral Entanglement  │   │  Capstone v258       │
│  Continuity  │   │  Reweighting Verifier     │   │  [claude/opus]       │
│  .258        │   │  (arXiv:2604.07650)       │   │  Cross-artifact      │
│  [NON-TERM]  │   │  [motivated by 2704 result]   synthesis            │
└──────────────┘   └───────────────────────────┘   └──────────────────────┘
```

---

## Phase Descriptions

### Phase A — Infrastructure + Activation (exp2712–exp2714)

**Goal:** Archive .257, activate .258, execute the P0 pre-test cascade fix, and ship Phase 1 prep artifacts.

**Why first:** The pre-test cascade is the single blocking issue for 51 milestones of research output. Fixing it in Position 2 of the pipeline means all Phase B–D tasks benefit. Gating downstream tasks on `exp2713.pretest_cascade_fixed == true` provides a clean skip path if the fix fails.

**exp2712 — Archive .257 + Activate .258:**
- Archives .257 (3 artifacts: exp2699, exp2700, exp2704; 10 absent) into research-complete.yaml.
- Copies research-roadmap-next.yaml → research-roadmap.yaml.

**exp2713 — Pre-Test Cascade Fix v1:**
- Installs `torch` CPU wheel in `.venv` via `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
- Patches `tests/python/inference/test_hw_dab.py`: replaces `import torch` with `import pytest; torch = pytest.importorskip('torch')` so the test gracefully skips if torch is absent rather than crashing pytest collection.
- Clears `ops/.pretest-cache.json` to force the conductor to re-evaluate the smart subset.
- Runs the smart subset to verify green (`81+ tests pass`).
- Does NOT modify `scripts/research_conductor.py` (operator must manually set `MAX_HEAL_ATTEMPTS = 1` as a follow-up after reviewing the 2026-05-03 quota-burn justification).
- Key output field: `pretest_cascade_fixed: bool` — gate for all Phase B–D tasks.

**exp2714 — Phase 1 Ship v5:**
- Executes remaining autonomous ship prep: README.md Phase 1 section, RELEASES.md, operator_ship_checklist_v5.
- Carry-forward from exp2701 (.257), exp2688 (.256), exp2674 (.255).

### Phase B — Live Evaluation + Verifier Stack (exp2715–exp2718)

**Goal:** First actual live GGUF pipeline validation + complete the verifier calibration stack.

**exp2715 — SOTA GGUF Live Pipeline Validation v3:**
- N=50 live GGUF inference on Qwen3.6-35B-A3B or Gemma-4-31B-it.
- Full PRECONDITIONS: GGUF cache check + CUDA check + FoVer corpus check.
- Primary output: `energy_score_distribution`, `inference_mode` ('live_gpu' / 'live_cpu' / 'smoke_only').
- MODEL_SPECS: unsloth/Qwen3.6-35B-A3B-GGUF (primary) + unsloth/gemma-4-31B-it-GGUF (secondary).

**exp2716 — Tier 0f Semantic Calibration v2:**
- Cluster-aware paraphrase calibration on FoVer corpus (arXiv:2605.15588).
- Carry-forward from exp2703.

**exp2717 — Property-Guided Counterexample Repair Loop v2:**
- ExVerus-style structured failure messages (arXiv:2603.25810) + property synthesis (arXiv:2605.16142).
- Adds `iterative_repair_with_counterexample()` to VerifyRepairPipeline.
- Carry-forward from exp2705.

**exp2718 — Linear Probe Calibration v2:**
- LinearProbeCalibrator on Tier 0e TF-IDF features (arXiv:2512.22245).
- 10x faster than multi-generation; measure ECE improvement.
- Carry-forward from exp2709.

### Phase C — FR-11 + Theory + New Research (exp2719–exp2721)

**Goal:** Complete FR-11 Tier 3 (mandatory), add ODAR routing research, update paper v6 theory.

**exp2719 — FR-11 Tier 3: ORCA TTT v2 + Grounded Continuation Stopping:**
- MANDATORY: FR-11 continuous self-learning mandate (`continuous_self_learning_task: true`).
- Creates `python/carnot/pipeline/ttt_loop.py` with VerifierDrivenTTT.
- ORCA conformal stopping (arXiv:2604.01170) + Grounded Continuation dual criterion (arXiv:2605.14175).
- Carry-forward from exp2706.

**exp2720 — ODAR Free-Energy Routing + T2 VegAS K-Scaling:**
- **NEW for .258:** ODAR (arXiv:2602.23681) free-energy routing — selects between fast-path and full verify-repair based on FEP-derived risk criterion. Directly motivated by Phase 4 active inference track.
- **Carry-forward T2 component (exp2707 scope):** T2 VegAS K-scaling laws with `ensemble_auroc=0.993` from exp2704. T2 predicts optimal K=3 (AUROC >= 0.85 threshold). Empirical verification.

**exp2721 — Paper v6 Theory Update v2:**
- Precondition-gated on pdflatex available AND .tex source found.
- ARM-EBM bijection §2 (arXiv:2512.15605, Score 500) + 4/δ bound §3 (arXiv:2512.02080) + FST §3 (arXiv:2605.12484).
- If toolchain absent: write `blocked_paper_v6_toolchain_or_source_missing` immediately (no fallback markdown).
- Carry-forward from exp2708.

### Phase D — Hardware + New Research + Capstone (exp2722–exp2724)

**exp2722 — KV260 Hardware Continuity .258:**
- MANDATORY per Hardware-Task Continuity Discipline (NON-TERMINAL).
- SD card absent in .254/.256/.257 (3 consecutive Branch B runs). This is the 4th.
- Branch A (SD present): xmutil loadapp + Ising energy smoke test.
- Branch B (SD absent): update hardware-bringup-prep.md noting 4 consecutive Branch B.

**exp2723 — Behavioral Entanglement Reweighting Verifier (arXiv:2604.07650):**
- **NEW for .258:** Motivated directly by exp2704's result (saturation_k=2, saturation_auroc=0.993, total_lift=-0.003). The negative total lift and early saturation are behavioral entanglement symptoms.
- Measures pairwise Pearson correlation of verifier scores on FoVer eval split.
- Applies de-entangled reweighting: weight each verifier by `1 - mean_pairwise_correlation`.
- Measures AUROC improvement under reweighting. Gate: `reweighted_auroc > 0.993`.

**exp2724 — Capstone v258 (claude/opus):**
- Cross-artifact synthesis of all .258 experiments.
- Updates ops/status.md, ops/changelog.md, ops/metrics.md.
- Produces `top_3_gaps_for_259` list.
- `requires_claude: true`: multi-file cross-artifact synthesis + doc updates across 12 upstream artifacts meets all 3 positive criteria.

---

## Dependency Graph

```
exp2712 ──▶ exp2713 ──┬──▶ exp2714
                      ├──▶ exp2715
                      ├──▶ exp2716
                      ├──▶ exp2717
                      ├──▶ exp2718
                      ├──▶ exp2719
                      ├──▶ exp2720
                      └──▶ exp2723

exp2704 (exp2704 data already in results/) ──▶ exp2720 (reads saturation_auroc=0.993)

exp2722: independent (hardware, no cascade dependency)
exp2721: independent (precondition-gated on pdflatex)
exp2724: reads all exp2712–exp2723 artifacts
```

---

## Hardware Requirements

| Experiment | Hardware | Precondition Gate |
|------------|----------|------------------|
| exp2715 | RTX 3090 CUDA + GGUF cache | cuda_available + gguf_cached |
| exp2719 | .venv/python only | carnot.pipeline importable |
| exp2720 | .venv/python only | carnot.pipeline importable |
| exp2721 | pdflatex + .tex source | pdflatex available |
| exp2722 | KV260 (USB-attached) | /dev/mmcblk* presence |
| exp2723 | .venv/sklearn only | sklearn installed |

---

## Agent Routing Summary

| Agent | Tasks | Count | % |
|-------|-------|-------|---|
| codex/gpt-5.5 | exp2712–exp2723 | 12 | 92.3% |
| claude/opus | exp2724 (capstone) | 1 | 7.7% |

- claude/opus ceiling: 2/13 = 15.4%. Actual: 1/13 = 7.7%. Well within ceiling.
- exp2724 meets all 3 positive criteria for `requires_claude: true`:
  1. Codex has failed multi-file synthesis capstones historically.
  2. 12+ file cross-artifact synthesis requires Read + Edit + Bash choreography.
  3. Open-ended judgment under ambiguity (some artifacts absent → synthesize from available evidence).

---

## CLAUDE.md Mandatory Disciplines — .258 Compliance Checklist

- [x] **Codex-Default (Quota Preservation):** 12/13 codex, 1/13 claude. Within 2/13 ceiling.
- [x] **Failed-Experiment Rerun Discipline:** All carry-forward tasks have `prior_failures:` with all 4 mandatory sub-fields (experiment_id, verdict, addressed_by, retire_if_same_verdict). New tasks (exp2713, exp2723) have `prior_failures: []`.
- [x] **Exclusion Manifest Cross-Check:** 0 scope matches against all retired IDs.
- [x] **Pre-Launch Preconditions:** All compute-bound tasks have PRECONDITIONS step 0.
- [x] **Principle-Annotated Artifact Fields:** All REQUIRED ARTIFACT FIELDS have `principle:` annotations.
- [x] **Verdict Terminal-Prefix Discipline:** All `honest_verdict` specs include the terminal-prefix requirement.
- [x] **FR-11 Mandate:** exp2719 (`continuous_self_learning_task: true`, ORCA TTT v2).
- [x] **Hardware-Task Continuity:** exp2722 (KV260 NON-TERMINAL).
- [x] **Operator-Only External Publication:** No submission steps in any task prompt.
- [x] **Never Stash — Always Commit-First:** Not applicable (planning only).
- [x] **SOTA GGUF Models:** exp2715 MODEL_SPECS includes Qwen3.6-35B-A3B-GGUF + gemma-4-31B-it-GGUF.
- [x] **Adversarial Artifact Verification:** All compute-bound tasks include `random_seed`, `duration_s`, `preconditions_checked`.
- [x] **Scope-Reduction:** No SCOPE REDUCTION directive active in ops/known-issues.md.

---

## Exclusion Manifest Cross-Check

Retired experiment IDs: 2091, 260, 308, 309, 346, 380, 381, 382, 383, 410, 425, 491, 527, 603, 627, 887, 783, 799, 804, 809, 825, 834, 872.
Retired scopes: HalluSAEGeometricProbe, GGUF-model-download-for-code-repair, discriminative JEPA OOD.

**Cross-check result: 0 scope matches.**

- Pre-test cascade fix: new scope (no prior experiment targeting this fix).
- GGUF Live Eval v3: uses transformers/llama.cpp loader (NOT the retired GGUF-model-download-for-code-repair scope which targeted CLI download for code repair).
- Behavioral Entanglement Reweighting: entirely new scope (arXiv:2604.07650), no prior attempt.
- All other carry-forwards: different experiment IDs from the retired list.

---

## Expected .258 Outcomes

| Outcome | Experiment | Gate |
|---------|-----------|------|
| Conductor pre-test cascade fixed | exp2713 | pretest_cascade_fixed == true |
| Phase 1 ship prep complete | exp2714 | phase1_ship_ready == true |
| First live GGUF pipeline validation | exp2715 | inference_mode in [live_gpu, live_cpu] |
| FR-11 Tier 3 (conformal TTT) | exp2719 | conformal_stopping_enabled == true |
| ODAR routing prototype | exp2720 | odar_routing_added == true |
| Behavioral entanglement reweighting | exp2723 | reweighted_auroc > 0.993 |
| KV260 continuity documented | exp2722 | branch_taken in [A, B] |
| Paper v6 theory cites | exp2721 | bijection_citation_added == true (if pdflatex available) |
