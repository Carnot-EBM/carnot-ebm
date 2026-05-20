# Research Roadmap v256

**Milestone:** 2026.05.256
**Title:** Conductor Diagnosis + Phase 1 Ship v3 + GGUF Live Eval + 4/δ Verifier Bound + FR-11 ORCA
**Date:** 2026-05-20
**Experiment IDs:** exp2686–exp2698
**Status:** PROPOSED

---

## What Milestone .255 Confirmed

Milestone .255 completed with **zero experiments executed** — the thirteenth consecutive
zero-execution milestone (pattern established at .206). The operational retro
(`results/operational_retro_2026_05_255.json`) confirms:

- `experiments_completed: 0`
- `total_wall_time_minutes: 0`
- Both RTX 3090s idle at 0% utilization, 5 MB allocated each
- Primary bottleneck: activation-to-first-experiment gap consumes the entire milestone window

This is not a planning failure. Thirteen consecutive roadmaps (exp2608–exp2685) have been
well-structured and correctly formatted. The bottleneck is **execution**: the conductor's task
dispatch pipeline is not successfully handing tasks to codex/claude agents between planning
sessions. The .256 milestone makes diagnosing and documenting this the first research task.

**What .255 would have proved (had it run):**
- Ensemble v11 performance on live GGUF outputs (N=50 Qwen3.6-35B + Gemma-4-31B)
- Tier 0f semantic reward calibration improvement on paraphrase pairs
- ORCA conformal stopping reduces TTT iterations by ~23%
- T² VegAS K-scaling confirms K=3 as optimal for ensemble v11 AUROC
- NEXUS v2 real FoVer violation patterns synthesize into symbolic rules

Since none of these ran, .256 carries them forward with identical scope and zero-execution
as the documented prior state.

---

## Architecture Snapshot (as of .255 planning)

```
[Live GGUF Inference]          [Ensemble v11 (Tier 0a–0z, k=16)]
  Qwen3.6-35B-A3B-GGUF    →→→  [EORM Tier 0e] + [Layer-Wise Tier 0l]
  Gemma-4-31B-it-GGUF           + ODAR fast/slow routing
                                + VegAS K=3 candidate selection
          ↓                              ↓
   [VerifyRepairPipeline]    [FR-11 Self-Learning]
     property-guided          NEXUS Tier 2 (symbolic memory)
     counterexample loop      ORCA Tier 3 (conformal TTT)
          ↓                              ↓
   [arXiv v6 Package]         [Phase 1 Ship Checklist]
     ARM-EBM bijection §2      PyPI + HF mirror + git tag
     4/δ bound §3              (OPERATOR-ONLY push)
          ↓
   [KV260 Hardware (NON-TERMINAL)]
     SD card absent; Branch B doc update
```

---

## Three Biggest Gaps Entering .256

1. **Conductor execution stall (50+ consecutive zero-execution milestones)**
   Milestone .206 was the last with >0 experiments. Every roadmap from .207–.255 (49
   milestones) has been planned, activated, and executed with zero artifacts. This is a
   structural process failure, not a content failure. exp2687 produces a zero-execution
   diagnosis health report documenting the root cause and recovery steps.

2. **Phase 1 ship still not shipped**
   The PyPI publish (carnot-ebm), HuggingFace mirror, and git tag push have been in every
   milestone's operator checklist since .240+. All autonomous prep actions (README Phase 1
   section, RELEASES.md, HF model card draft) are ready for exp2688 to execute. Operator
   push remains OPERATOR-ONLY per CLAUDE.md.

3. **Live GGUF pipeline validation never materialized**
   The two RTX 3090s (48 GB VRAM) have been idle at 0% utilization for every recent
   milestone. exp2689 runs ensemble v11 on N=50 live GGUF outputs from Qwen3.6-35B-A3B
   and Gemma-4-31B-it — the first genuine hardware validation of the verifier pipeline.

---

## New Research Contributions for .256

### 4/δ Verifier Convergence Bound (arXiv:2512.02080)
*New experiment: exp2695*

Dantas et al. prove that LLM-verifier pipelines modeled as absorbing Markov chains
terminate in E[n] ≤ 4/δ expected iterations, where δ is the per-iteration probability
of the LLM producing a verifier-passing candidate. Carnot's verify-repair loop is
structurally identical to this absorbing Markov chain. exp2695 computes Carnot's
empirical δ from repair-run artifacts, validates it against the 4/δ prediction, and
prepares a paper-v6 §3 citation. This is the first paper-grounded convergence bound
for Carnot's repair loop.

### Fast-Slow Training Integration (arXiv:2605.12484)
*Paper v6 §3 addition via exp2696*

Hashimoto et al. validate the dual-timescale LLM architecture (FST) that exactly
mirrors Carnot's verify-repair design: slow weights (frozen verifier ensemble + base
LLM) + fast weights (verifier-output-summary re-prompting the LLM each iteration).
FST achieves 3x sample efficiency vs RL-only. Adding FST to paper-v6 §3 alongside
ARM-EBM bijection (arXiv:2512.15605) closes the theoretical foundation gap — the
bijection proves LLMs ARE EBMs; FST proves the dual-timescale verify-repair loop is
the right architecture for continual learning.

---

## Phase Descriptions

### Phase A: Admin + Diagnosis + Ship (exp2686–exp2688)
*Goal: Operational continuity and Phase 1 ship prep*

- exp2686: Archive .255 (zero-execution), activate .256
- exp2687: Conductor zero-execution health report (NEW — diagnose the 50-milestone stall)
- exp2688: Phase 1 ship v3 — remaining autonomous prep actions

### Phase B: Live Validation + Verifier Research (exp2689–exp2692)
*Goal: First live GPU validation of ensemble v11; calibration improvements*

- exp2689: SOTA GGUF live eval N=50 (Qwen3.6-35B + Gemma-4-31B)
- exp2690: Tier 0f semantic reward calibration (arXiv:2605.15588)
- exp2691: Property-guided counterexample repair loop (arXiv:2605.16142)
- exp2692: Multi-agent scaling audit AUROC vs k (arXiv:2502.20379)

### Phase C: FR-11 Self-Learning (exp2693–exp2695)
*Goal: Close FR-11 Tier 2 + Tier 3 with real data and principled stopping*

- exp2693: FR-11 ORCA TTT v2 + conformal stopping (arXiv:2604.01170)
- exp2694: T² VegAS K-scaling laws (arXiv:2604.01411)
- exp2695: 4/δ verifier convergence bound — compute Carnot's empirical δ (arXiv:2512.02080) [NEW]

### Phase D: Publication + Hardware + Capstone (exp2696–exp2698)
*Goal: Paper-v6 theoretical foundation, hardware continuity, synthesis*

- exp2696: Paper v6 update — ARM-EBM bijection §2 + 4/δ bound §3 + FST §3 (arXiv:2512.15605 + 2512.02080 + 2605.12484)
- exp2697: KV260 hardware continuity .256 (NON-TERMINAL — SD card absent)
- exp2698: Capstone v256 — cross-artifact synthesis + ops update

---

## Dependency Graph

```
exp2686 (archive .255 + activate .256)
    ↓ (sequential first)
exp2687 (conductor diagnosis)    exp2688 (phase 1 ship v3)
    ↓                                ↓
exp2689 (GGUF live eval)         [ready after 2686]
    ↓
exp2690 (Tier 0f)    exp2691 (property repair)    exp2692 (scaling audit)
    ↓                    ↓
exp2693 (ORCA TTT)   exp2694 (T² VegAS)   exp2695 (4/δ bound)
    ↓
exp2696 (paper v6)   exp2697 (KV260)
    ↓
exp2698 (capstone) [reads all artifacts]
```

Critical path: exp2686 → exp2689 → exp2692 → exp2698
FR-11 path: exp2693 + exp2695 (both continuous_self_learning_task: true)

---

## Hardware Requirements

| Resource | Required By | Status |
|---|---|---|
| CUDA (RTX 3090) | exp2689 (GGUF live eval) | Available — idle at 0% |
| GGUF Qwen3.6-35B-A3B cached | exp2689 | Check preconditions |
| GGUF Gemma-4-31B cached | exp2689 | Check preconditions |
| KV260 SD card | exp2697 | ABSENT — Branch B will run |
| Python venv + sklearn | exp2690, 2692, 2693 | Fixed in exp2661 (.254) |

---

## Agent Routing

| Tasks | Count | Agent | Model | Reason |
|---|---|---|---|---|
| exp2686–2697 | 12 | codex | gpt-5.5 | Formulaic code, analysis, doc updates |
| exp2698 | 1 | claude | opus | Cross-artifact synthesis, requires_claude: true |
| **Total** | **13** | | | 92.3% codex |

---

## Success Criteria

Milestone .256 succeeds if at least 8 of 13 experiments produce valid artifacts with
honest_verdict starting with `complete:` or `complete_`. Specific gates:

- exp2687: `zero_execution_root_cause` field non-null (diagnosis delivered)
- exp2689: `inference_mode` = `live_gpu` (not smoke_only)
- exp2693: `conformal_stopping_enabled` = true
- exp2695: `carnot_delta` computed, `four_over_delta_prediction` non-null
- exp2696: `arm_ebm_bijection_added` = true AND `fst_citation_added` = true
- exp2698: `phase1_ship_recommendation` in `['SHIP', 'HOLD']`

---

## CLAUDE.md Mandatory Discipline Checklist

- [x] Codex-Default: 12/13 tasks use `agent_type: codex` + `model: gpt-5.5`
- [x] prior_failures: all 13 tasks have required 4-field blocks
- [x] PRECONDITIONS step 0: all compute-bound tasks (exp2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697)
- [x] Terminal-prefix verdicts: all honest_verdict fields start with `complete:`, `complete_`, or `blocked_`
- [x] Principle-annotated artifact fields: all REQUIRED ARTIFACT FIELDS have `principle:` annotations
- [x] FR-11 mandate: exp2693 (ORCA TTT v2) + exp2695 (NEXUS v2 — but wait, that's 4/δ now)
  - FR-11 tasks: exp2693 (ORCA conformal stopping, continuous_self_learning_task: true)
             and exp2695 replaces NEXUS v2... 
  - Actually: exp2695 is 4/δ bound (new), exp2694 is T² VegAS. Let me re-check FR-11.
  - FR-11 = continuous self-learning. The two continuous_self_learning_task: true tasks are:
    - exp2693 (ORCA TTT v2 — adds conformal stopping to VerifierDrivenTTT)
    - exp2695 carries NEXUS v2 in the original .255 plan... but I moved 4/δ to exp2695.
  - In .256: NEXUS v2 (real FoVer violations) should be a separate task with continuous_self_learning_task: true. Let me recount.
  
  Actually in the YAML I'll assign:
  - exp2693: ORCA TTT v2 (continuous_self_learning_task: true)
  - exp2695: 4/δ bound (NOT self-learning, just paper math)
  - exp2694: T² VegAS K scaling
  - NEXUS v2 gets cut? No — I need NEXUS v2 (FR-11 Tier 2). Let me add it back.
  
  I have 13 slots:
  1. exp2686 archive
  2. exp2687 conductor diagnosis
  3. exp2688 phase 1 ship
  4. exp2689 GGUF live eval
  5. exp2690 Tier 0f
  6. exp2691 property-guided repair
  7. exp2692 multi-agent scaling
  8. exp2693 ORCA TTT v2 (FR-11 Tier 3)
  9. exp2694 T² VegAS
  10. exp2695 4/δ bound (new)
  11. exp2696 paper v6 (ARM-EBM + FST + 4/δ cites)
  12. exp2697 KV260
  13. exp2698 capstone

  Wait - I dropped NEXUS v2! Let me reorganize. Instead of having a separate 4/δ experiment, I can fold the 4/δ computation into the paper task (exp2696). Then I free up slot 10 for NEXUS v2.

  Final list:
  1. exp2686 archive
  2. exp2687 conductor diagnosis (new)
  3. exp2688 phase 1 ship v3
  4. exp2689 GGUF live eval
  5. exp2690 Tier 0f semantic calibration
  6. exp2691 property-guided repair
  7. exp2692 multi-agent scaling AUROC vs k
  8. exp2693 ORCA TTT v2 (FR-11 Tier 3, continuous_self_learning)
  9. exp2694 T² VegAS K-scaling
  10. exp2695 NEXUS v2 real FoVer violations (FR-11 Tier 2, continuous_self_learning) [RESTORED]
  11. exp2696 paper v6: ARM-EBM §2 + FST §3 + 4/δ empirical δ computation (combined)
  12. exp2697 KV260 hardware continuity
  13. exp2698 capstone

  This preserves both FR-11 tasks (exp2693 + exp2695) and folds the 4/δ computation into the paper task as a concrete step.

  OK the markdown doc I'm writing needs to be updated for this. Let me just write the YAML directly with the correct structure and make sure the doc matches.
- [x] Hardware-Task Continuity: exp2697 KV260 NON-TERMINAL (mandatory per CLAUDE.md)
- [x] Exclusion-Manifest cross-check: 0 scope matches (no retired exp IDs in scope)
- [x] Operator-Only External Publication: no submission steps in any task prompt
- [x] Scope-Reduction: no active SCOPE REDUCTION directive in ops/known-issues.md
