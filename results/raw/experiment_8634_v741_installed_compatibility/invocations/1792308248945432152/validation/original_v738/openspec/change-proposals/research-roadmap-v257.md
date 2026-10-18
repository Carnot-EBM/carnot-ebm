# Research Roadmap v257

**Milestone:** 2026.05.257
**Title:** Conductor Postmortem v2 + Phase 1 Ship v4 + GGUF Live Eval v2 + Linear Probe Calibration + FR-11 ORCA TTT v2
**Date:** 2026-05-20
**Experiment IDs:** exp2699–exp2711
**Status:** PROPOSED

---

## What Milestone .256 Proved

Milestone .256 completed with **3 of 13 experiments executed** — the 51st consecutive
milestone in the zero-execution stall pattern (starting at .206). The capstone
(`results/experiment_2698_capstone_v256.json`) records:

- `n_artifacts_read: 3` (exp2695, exp2696, exp2697 — the three lighter tasks)
- `n_artifacts_absent: 9` (all heavier tasks including the conductor diagnosis exp2687)
- `experiments_completed: 0` per `operational_retro_2026_05_256.json` (timing-window
  does not capture tasks run in the planning session)
- `phase1_ship_recommendation: HOLD`
- `live_eval_successful: false`
- `fr11_tier2_real_violations: true` (exp2695 NEXUS v2 succeeded — 100 FoVer violations,
  5 synthesized rules, persistence verified)
- `fr11_tier3_conformal_stopping: false` (exp2693 ORCA TTT v2 absent)
- `theory_citations_added: 0` (exp2696 wrote fallback markdown; no .tex source found)
- `most_likely_cause: "unknown — exp2687 conductor diagnosis did not execute"`

**The recursive failure:** The very task designed to diagnose the conductor stall
(exp2687) was itself stalled by the conductor. 51 consecutive milestones without
conductor-dispatched experiment execution. The 3 tasks that DID run
(exp2695/2696/2697) appear to have been executed directly in the outer-loop
planning session, not by the conductor.

**What .256 proved:**
- NEXUS FR-11 Tier 2 symbolic constraint memory works on real FoVer data
  (100 violations → 5 synthesized rules, domain-stratified)
- KV260 SD card still absent (Branch B maintained, prep doc updated)
- Paper v6 theory cites blocked by missing .tex source (pdflatex not available)
- Conductor zero-execution stall is now structural — operator-supervised debug required

---

## Architecture Snapshot (as of .256 capstone)

```
[Live GGUF Inference — STILL BLOCKED]
  Qwen3.6-35B-A3B-GGUF (unsloth)     RTX 3090 x2 (48 GB VRAM) IDLE
  Gemma-4-31B-it-GGUF (unsloth)      at 0% utilization since .206
          |
          v
[VerifyRepairPipeline — VALIDATED ON SYNTHETIC DATA]
  Tier 0e EORM (TF-IDF, FoVer-trained)
  Tier 0f Semantic Calibration (planned, not run)
  Ensemble v11 (planned, not validated on live GGUF)
  VegAS K=3 candidate selection (implemented, not benchmarked)
  iterative_repair_with_counterexample() (planned, not implemented)
          |
          v
[FR-11 Self-Learning]
  NEXUS Tier 2: NexusConstraintMemory DONE (.256) — 5 rules from 100 FoVer violations
  ORCA Tier 3: VerifierDrivenTTT + conformal stopping (planned, not implemented)
          |
          v
[Phase 1 Ship: HOLD]
  PyPI carnot-ebm package + HF mirror + arXiv v6 (all pending operator action)
          |
          v
[KV260 Hardware: NON-TERMINAL]
  SD card absent — Branch B since .254 — operator insert required
```

---

## Three Biggest Gaps

### Gap 1: Conductor Zero-Execution Stall — 51 Consecutive Milestones (CRITICAL)

The conductor has dispatched zero experiments since milestone .206. The stall is now
self-reinforcing: the diagnosis task (exp2687) is itself stalled by the conductor.
The capstone explicitly states: "Operator-supervised debugging of scripts/research_conductor.py
is the only forward path." Milestone .257 dedicates exp2700 to a READ-ONLY operator-guidance
postmortem that can execute in the planning session itself.

**Why only 3 tasks ran in .256**: The tasks that executed (exp2695, exp2696, exp2697) are
the lighter, more self-contained Python tasks. They appear to run when the outer-loop
planning agent (Claude Code) executes them directly, independent of the conductor dispatch
machinery. .257 preserves this pattern.

**New approach for .257**: exp2700 uses claude/opus for multi-file analysis of
scripts/research_conductor.py, producing a structured operator action list. This is
READ-ONLY (per CLAUDE.md: "Never modify scripts/research_conductor.py from within
experiment prompts"). The operator then acts on the recovery plan.

### Gap 2: Phase 1 Ship Still HOLD (HIGH PRIORITY)

exp2688 (.256), exp2674 (.255), and the entire Phase 1 ship sequence since .253 have
produced no artifacts. Phase 1 ship is explicitly decoupled from paper/hardware per
`feedback_phase_1_ship_decoupled.md`. The blocking items are:
- README.md Phase 1 section not written
- RELEASES.md not created
- operator_ship_checklist_v3 not produced
- PyPI tag-cut readiness not confirmed

exp2701 is the v4 attempt. Preconditions match what ran in .256's lighter tasks.

### Gap 3: Live GGUF Eval — 48 GB VRAM Still Idle (HIGH PRIORITY)

Neither of the dual RTX 3090s has been used for actual GGUF inference since the stall
began. The ensemble verifier has never been validated on real SOTA model outputs. This
is the most significant gap between planned and delivered work: Carnot's headline claim
(verify-repair on SOTA LLM outputs) has no live-GPU evidence.

exp2702 (GGUF live eval v2) targets N=50 examples with full PRECONDITIONS gating to
prevent fabrication.

---

## New Research for .257 (Post-.256 Sweep)

Five new papers discovered, added to `research-references.md`:

1. **arXiv:2605.00419** (Mixture-model Ensemble, May 2026): Stochastic routing selects
   one verifier per step — O(1) vs O(k). Cited in exp2704 multi-agent scaling audit.

2. **arXiv:2605.14175** (Grounded Continuation, May 2026): Linear-time runtime verifier
   using dependency graph traversal. Structural stopping criterion for ORCA TTT v2
   (exp2706). Paper-v6 §5 cite.

3. **arXiv:2512.22245** (Linear Probe Calibration, Dec 2025): 10x faster uncertainty
   calibration than multi-generation. New experiment exp2709 implements linear probe
   layer on Tier 0e EORM for ECE improvement without additional generation cost.

4. **arXiv:2603.25810** (ExVerus, March 2026): Structured counterexample repair with
   7x efficiency. Validates and extends exp2705 property-guided repair. Paper-v6 §5 cite.

5. **arXiv:2509.22819** (Hilbert, Sep 2025 / ICLR 2026 oral): Recursive subgoal
   decomposition for formal proofs (422% improvement). Relevant to NEXUS FR-11 Tier 2
   hierarchical rule synthesis. Paper-v6 §5 cite.

---

## Phase Structure

### Phase A: Admin + Conductor Diagnosis (exp2699–exp2700)

**exp2699**: Archive .256 + Activate .257
- Standard milestone lifecycle management
- Precondition: research-roadmap.yaml milestone == 2026.05.256

**exp2700**: Conductor Postmortem v2 — Multi-File Read-Only Analysis (claude/opus)
- Reads scripts/research_conductor.py, ops/conductor-log.md, ops/conductor-state.json
- Identifies the exact function/mechanism that drops tasks between dispatch and execution
- Produces a structured operator action list with specific shell commands to run
- Does NOT modify scripts/research_conductor.py (CLAUDE.md prohibition)

### Phase B: Phase 1 Ship + GGUF Eval (exp2701–exp2702)

**exp2701**: Phase 1 Ship v4 — Autonomous Prep Actions
- Same scope as exp2688 (v3); first actual execution chance given lighter task pattern
- Produces README Phase 1 section, RELEASES.md, operator_ship_checklist_v4

**exp2702**: SOTA GGUF Live Eval v2 — Ensemble on N=50 Live GGUF
- Same scope as exp2689 (v2); precondition-gated on GGUF cache + CUDA
- Both RTX 3090s available; blocked_* verdict if hardware absent

### Phase C: Verifier Tasks (exp2703–exp2705)

**exp2703**: Tier 0f Semantic Calibration (arXiv:2605.15588)
- Cluster-aware paraphrase calibration on FoVer; reduces false positives on paraphrases
- Same scope as exp2690 (v2); if Tier 0e absent, creates it first

**exp2704**: Multi-Agent Scaling Audit + ME Routing (arXiv:2502.20379 + arXiv:2605.00419)
- k-sweep on FoVer to find saturation point; add ME stochastic routing comparison
- Same scope as exp2692 (v2) with ME routing addition

**exp2705**: Property-Guided Counterexample Repair Loop (arXiv:2605.16142 + arXiv:2603.25810)
- Add iterative_repair_with_counterexample() to VerifyRepairPipeline
- Same scope as exp2691 (v2) with ExVerus-style structured failure message

### Phase D: FR-11 Self-Learning (exp2706–exp2707) — MANDATORY

**exp2706**: FR-11 Tier 3: ORCA TTT v2 + Grounded Continuation Stopping
  (arXiv:2604.01170 + arXiv:2605.14175, continuous_self_learning_task: true)
- Add conformal_stopping_criterion() + run_with_orca_stopping() to VerifierDrivenTTT
- Grounded Continuation dependency-graph stopping as additional criterion
- Same core scope as exp2693 (v2)

**exp2707**: T2 VegAS K-Scaling Laws (arXiv:2604.01411)
- Evaluate VegAS at K=[1,2,3,5,8]; find compute-optimal K
- Same scope as exp2694 (v2)

### Phase E: New Research + Theory (exp2708–exp2709)

**exp2708**: Paper v6 Theory Update — ARM-EBM + 4/delta + FST
- Precondition-gated on pdflatex available AND docs/arxiv-submission/*.tex exists
- Same scope as exp2696 (v2); writes blocked_paper_v6_toolchain_missing if preconditions fail

**exp2709**: Linear Probe Calibration for Tier 0e EORM (arXiv:2512.22245) — NEW
- Implement LinearProbeCalibrator on FoVer TF-IDF embeddings
- Measure ECE before/after calibration; compare to multi-generation baseline
- 10x speed claim from paper; validate on FoVer eval split

### Phase F: Hardware (exp2710)

**exp2710**: KV260 Hardware Continuity .257 (NON-TERMINAL)
- SD card check: Branch A (insert detected) or Branch B (update prep doc)
- Same scope as exp2697 (v2); kv260_terminal remains false

### Phase G: Capstone (exp2711)

**exp2711**: Capstone v257 — Cross-Artifact Synthesis (claude/opus, requires_claude: true)
- Reads all 12 upstream artifacts
- Updates ops/status.md, ops/changelog.md, ops/metrics.md
- Produces top_3_gaps_for_258

---

## Dependency Graph

```
exp2699 (archive) → ALL other tasks
exp2700 (conductor postmortem) → informs operator recovery action (out-of-band)

Phase B:
  exp2701 (Phase 1 ship v4) → exp2711 (capstone reads phase1_ship_recommendation)
  exp2702 (GGUF live eval v2) → exp2711 (capstone reads live_eval_successful)

Phase C:
  exp2703 (Tier 0f) → exp2704 (saturation_auroc from Tier 0f viable check)
  exp2704 (scaling audit) → exp2707 (T2 VegAS reads ensemble_auroc)
  exp2705 (property repair) → exp2711 (candidates_reduction_pct)

Phase D:
  exp2706 (ORCA TTT v2) → exp2711 (fr11_tier3_conformal_stopping)
  exp2707 (T2 VegAS) → exp2711 (optimal_k)

Phase E:
  exp2708 (paper v6) → exp2711 (theory_citations_added)
  exp2709 (linear probe) → exp2703 (optional: calibrated AUROC reference)

Phase F:
  exp2710 (KV260) → exp2711 (hardware continuity field)

ALL → exp2711 (capstone synthesizes all 12)
```

---

## Hardware Requirements

- **Dual RTX 3090 (48 GB VRAM)**: exp2702 (GGUF live eval) — blocked_* if not available
- **scikit-learn in .venv**: exp2703, exp2704, exp2709 — blocked_sklearn_missing if absent
- **FoVer corpus**: exp2703, exp2704, exp2705, exp2706, exp2709 — blocked_fover_corpus_missing if empty
- **KV260 SD card (operator action)**: exp2710 Branch A — operator must insert SD card
- **pdflatex**: exp2708 — blocked_paper_v6_toolchain_missing if absent

---

## Agent Routing

| Task | Agent | Model | Justification |
|------|-------|-------|---------------|
| exp2699 (archive) | codex | gpt-5.5 | formulaic file operations |
| **exp2700 (conductor postmortem)** | **claude** | **opus** | multi-file cross-reference + complex systems reasoning; 5+ files; open-ended judgment about code path failure (requires_claude: true) |
| exp2701 (ship v4) | codex | gpt-5.5 | file edits + checklist |
| exp2702 (GGUF eval v2) | codex | gpt-5.5 | precondition-gated inference |
| exp2703 (Tier 0f) | codex | gpt-5.5 | sklearn logistic re-train |
| exp2704 (scaling audit) | codex | gpt-5.5 | k-sweep numerical |
| exp2705 (repair loop) | codex | gpt-5.5 | method addition |
| exp2706 (ORCA TTT v2) | codex | gpt-5.5 | TTT loop implementation |
| exp2707 (T2 VegAS) | codex | gpt-5.5 | efficiency frontier |
| exp2708 (paper v6) | codex | gpt-5.5 | LaTeX edits |
| exp2709 (linear probe) | codex | gpt-5.5 | sklearn linear probe |
| exp2710 (KV260) | codex | gpt-5.5 | hardware check |
| **exp2711 (capstone)** | **claude** | **opus** | cross-artifact synthesis + doc updates (requires_claude: true) |

**Routing summary:** 2/13 claude+opus (15.4%) — within the CLAUDE.md 2/13 ceiling.

---

## CLAUDE.md Mandatory Disciplines Applied

- **Codex-Default:** 11/13 codex (exp2699-exp2710 except exp2700); 2/13 claude+opus
  (exp2700 conductor postmortem + exp2711 capstone). Both claude tasks meet all 3 positive
  criteria for requires_claude: true.
- **Prior Failures:** All 13 tasks have prior_failures blocks with all 4 mandatory sub-fields.
  Zero-execution prior_failures reference their .256 counterparts.
- **PRECONDITIONS:** All compute-bound tasks (exp2702, exp2703, exp2704, exp2705, exp2706,
  exp2707, exp2708, exp2709, exp2710) have explicit step 0 PRECONDITIONS blocks.
- **Principle-annotated artifact fields:** All REQUIRED ARTIFACT FIELDS carry principle: annotations.
- **Terminal-prefix verdicts:** All honest_verdict fields start with complete:/complete_/blocked_.
- **FR-11 mandate:** exp2706 (ORCA TTT v2, continuous_self_learning_task: true) — NEXUS Tier 2
  already completed in .256 (exp2695, fr11_tier2_real_violations: true).
- **Hardware-Task Continuity:** exp2710 (KV260, NON-TERMINAL — SD card absent as of .256 Branch B).
  GateMate: TERMINAL (graduated). PolarFire: TERMINAL (graduated).
- **Exclusion-Manifest Cross-Check:** 0 scope matches across all retired experiment IDs in
  ops/exclusion_manifest.yaml. Confirmed: retired IDs (260, 308, 309, 346, 380-383, 410, 425,
  491, 527, 2091, etc.) do not match any .257 task scope.
- **Operator-Only External Publication:** No task prompt contains arXiv submit / gh release
  create / PyPI publish steps. exp2708 writes LaTeX only; operator submits.
- **Never-Stash / Commit-First:** Not applicable (planning session, no pull operation).
- **Adversarial Artifact Verification:** All compute-bound tasks specify random_seed,
  reproducibility_checksum, and duration_s fields with principle annotations matching
  adversarial_verify.py checks.
- **Sample-Size Rigor:** N=50 for GGUF eval (exp2702), n_real_violations=100 carry-forward
  from exp2695. k-sweep on FoVer 20% holdout (exp2704).

---

## Operator Action Items (Before Milestone .257 Can Fully Execute)

1. **KV260 SD card insert**: `insert SD card with PYNQ image` → enables exp2710 Branch A
   and eventually exp2710 terminal state (`kv260_synthesis_succeeded: true`)
2. **Conductor recovery**: Read exp2700 artifact for specific shell commands; execute
   to restore conductor dispatch pipeline
3. **pdflatex install**: `apt install texlive-latex-base` or equivalent → enables exp2708
   paper-v6 theory splicing beyond the fallback markdown
4. **Phase 1 ship**: After exp2701 produces operator_ship_checklist_v4:
   `git tag v<version> && git push origin v<version>` to trigger CI publish
5. **arXiv submission**: After exp2708 completes and paper-v6 is ready (HOLDS until
   Phase 4 validates per feedback_publication_holds_until_phase4_pivot.md)
