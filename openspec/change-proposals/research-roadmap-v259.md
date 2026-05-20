# Carnot Research Roadmap — Milestone 2026.05.259

**Milestone:** 2026.05.259
**Title:** Verifier Energy Debug v1 + Full Test Fix + pdflatex Install + OTV Fast Path + FR-11 ORCA-NEXUS
**Status:** PROPOSED
**Date:** 2026-05-20
**Experiment IDs:** exp2725–exp2737

---

## What Milestone .258 Proved

Milestone 2026.05.258 was the **first fully-executed milestone** since the conductor cascade stall that
began at milestone .206. All 13 of 13 tasks produced artifacts (8 of 12 acceptance criteria met).

Key findings from .258:

1. **Cascade fixed** (exp2713): torch CPU wheel installed + graceful importorskip patch + pretest cache
   cleared. `pretest_cascade_fixed: true`, `smart_subset_passes: true`. **However**: `full_collection_clean:
   false` — there are OTHER test collection failures beyond test_hw_dab.py that remain unresolved.

2. **Phase 1 ship ready** (exp2714): `phase1_ship_ready: true`. README Phase 1 section already present.
   RELEASES.md created. `operator_ship_checklist_v5` complete. Capstone recommends "SHIP". Operator
   action: `git tag v0.1.0b1 && git push origin main v0.1.0b1` triggers CI → PyPI.

3. **Live GGUF eval DEGENERATE** (exp2715): Ran 50 examples via Qwen3.6-35B-A3B-GGUF on RTX 3090
   (CUDA available, 125s wall clock), BUT `energy_score_distribution=all-zero` (mean=std=min=max=0.0).
   The verifier pipeline is non-discriminative on live GGUF outputs. This is the most critical finding
   for .259 — the core verification capability does not produce meaningful scores on real model outputs.

4. **Paper v6 toolchain blocked** (exp2721): `pdflatex_available: false` — texlive is not installed.
   Theory citations (ARM-EBM bijection + 4/delta + FST) cannot be added until toolchain is installed.
   `carnot_delta: 0.25` was computed conservatively but not landed.

5. **Behavioral entanglement reweighting hypothesis NOT confirmed** (exp2723): `auroc_lift: -0.008`
   (NEGATIVE), `high_entanglement: false`. The verifier ensemble is NOT behaviorally entangled by the
   pairwise Pearson correlation criterion. De-entangled reweighting made performance slightly worse.

6. **Linear probe speedup implausible** (exp2718): `speedup_factor: 1.85e6`. Adversarial-verify
   flagged this as DURATION_TOO_SHORT equivalent — the baseline was simulated time (n_eval × 2s),
   not measured time. Needs replication with proper instrumentation.

7. **FR-11 Tier 3 ORCA TTT v2 operational** (exp2719): `conformal_stopping_enabled: true`,
   `n_ttt_steps_saved: 79`. Both ORCA conformal stopping and Grounded Continuation dependency-graph
   stopping are now operational.

8. **ODAR routing added** (exp2720): `odar_routing_added: true`, `t2_prediction_matches: true`.
   Free-energy-principled routing is now in the pipeline. T2 optimal K confirmed: K=3.

9. **Tier 0f semantic calibration viable** (exp2716): `tier0f_auroc: 0.992`, `tier0f_viable: true`.

10. **KV260 SD card absent** (exp2722): 4th consecutive Branch B.

---

## Three Biggest Gaps for .259

### Gap 1: Live GGUF Verifier Produces Zero Energy Scores (CRITICAL)

`exp2715.energy_score_distribution = {mean: 0.0, std: 0.0, min: 0.0, max: 0.0}` on 50 real
Qwen3.6-35B-A3B-GGUF outputs. The verifier pipeline runs (125s, `inference_mode: live_gpu`) but
`VerifyRepairPipeline.verify()` outputs zero for every example. This means:
- Carnot's core capability (verify LLM outputs) does NOT work on state-of-the-art GGUF models
- All claims about verifier AUROC are based on FoVer corpus synthetic data, not live model outputs
- Phase 1 ship is technically "ready" but the product ships with a degenerate verifier

**Root cause hypotheses:**
- `should_verify()` fast-path always returns False (fast_path_rate=0.0 contradicts this)
- `verify()` is computing energy=0 for all inputs (possible if FoVer features don't transfer to GGUF output format)
- GGUF response format (markdown, special tokens, long context) differs from FoVer training distribution
- TF-IDF features learned on FoVer corpus do not transfer to Qwen3.6-35B response style

**Fix strategy:**
1. Add diagnostic logging to VerifyRepairPipeline.verify() to trace energy computation steps
2. Compare FoVer training examples vs Qwen3.6-35B response format
3. Test with Tier 0a (simplest verifier) directly to isolate which tier collapses to zero
4. NEW: Semantic Energy (arXiv:2508.14496) provides a principled alternative that computes energy
   from semantic clustering rather than learned features — more robust to distribution shift.

### Gap 2: Paper v6 Toolchain Blocked (pdflatex not installed)

Three consecutive milestones have failed to add theory citations because pdflatex is not on PATH.
The fix is mechanical: `apt install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended`.
Once pdflatex is available, exp2729 can:
1. Find or create the .tex source tree (ARM-EBM bijection §2, 4/delta §3, FST §3)
2. Add bibliography entries for arXiv:2512.15605, arXiv:2512.02080, arXiv:2605.12484
3. Compile and verify LaTeX compiles clean

### Gap 3: Full Test Collection Not Clean (full_collection_clean: false)

exp2713 fixed the smart-subset tests but `full_collection_clean: false` means pytest discovers
failures in `tests/python/ --co -q`. If any new test file imports a missing package without
`importorskip`, a future git diff including that file could re-trigger the cascade. The risk
window is small (smart_subset is narrowed), but elimination is cleaner.

---

## New Research Opportunities

From the post-.258 arxiv sweep (4 new papers added to research-references.md):

1. **arXiv:2508.14496 (Semantic Energy)** — Addresses Gap 1 directly. Boltzmann energy over semantic
   clusters is non-degenerate by design (cluster probabilities are never all-zero). Candidate Tier 0g.

2. **arXiv:2603.01025 (OTV — One-Token Verification)** — Ultra-cheap fast-path routing via KV-cache
   probe. 90% token reduction. Two-tier fast path: OTV → ODAR → full ensemble. Complements exp2720.

3. **arXiv:2602.01090 (FALCON)** — Hard constraints + soft generation separation mirrors Carnot's
   two-layer architecture. Paper-v6 §5 peer cite. Validates exp2717 counterexample repair design.

4. **arXiv:2506.03723 (Verbalized Confidence)** — Alternative Phase 4 mechanism: verbalize confidence
   as routing signal. Simpler than ODAR, possibly more reliable. Paper-v6 §6 candidate.

---

## Architecture After .259

```
VerifyRepairPipeline (post-.259 target)
│
├── OTV Fast Path (exp2728) ─── KV-cache probe, 90% token reduction
│   ├── Low uncertainty → RETURN (fast path)
│   └── High uncertainty → ODAR routing (exp2720)
│                            ├── Fast path (F < threshold) → SKIP verify
│                            └── Deliberative path → Ensemble verify
│
├── Verifier Ensemble (post-fix from exp2727 debug)
│   ├── Tier 0a–0z (k=16 base verifiers)
│   ├── Tier 0f SemanticCalibratedVerifier (exp2716)
│   ├── Tier 0g Semantic Energy Verifier (exp2731) [NEW]
│   └── De-entangled weights (retired in exp2732 — ensemble diversity selection instead)
│
├── Repair Loop (exp2717 iterative + FALCON integration exp2734)
│   └── ExVerus counterexample format + FALCON grammar-constrained sampling
│
└── FR-11 Self-Learning (exp2733 ORCA-NEXUS Integration)
    ├── NEXUS symbolic constraint memory (exp2695 — completed)
    ├── ORCA conformal stopping (exp2719 — completed)
    └── Online violation learning (exp2733 — .259)
```

---

## Phase Structure

### Phase A: Admin + Infrastructure (exp2725–exp2726)
- exp2725: Archive .258 + Activate .259 (20 turns, codex)
- exp2726: Full Test Suite Collection Fix v1 (30 turns, codex)

### Phase B: Critical — Verifier Energy Debug + Fast Path (exp2727–exp2728)
- exp2727: Live GGUF Verifier Energy Debug v1 (50 turns, codex) — traces verify() zero-energy bug
- exp2728: OTV One-Token Verification Fast Path (arXiv:2603.01025) (35 turns, codex)

### Phase C: Paper v6 + Phase 1 Ship (exp2729–exp2730)
- exp2729: pdflatex Install + Paper v6 Theory v3 (35 turns, codex, prior_failures from exp2721)
- exp2730: HuggingFace Mirror Prep + Phase 1 Ship v6 Final (25 turns, codex)

### Phase D: Research — New Verifiers + Ensemble + Self-Learning (exp2731–exp2734)
- exp2731: Semantic Energy Verifier Tier 0g (arXiv:2508.14496) (40 turns, codex)
- exp2732: Behavioral Entanglement Lineage Retirement + Diversity Audit (25 turns, codex)
- exp2733: FR-11 ORCA-NEXUS Integration v1 (continuous_self_learning_task: true) (40 turns, codex)
- exp2734: FALCON Property-Constrained Repair Integration (arXiv:2602.01090) (35 turns, codex)

### Phase E: Hardware + Publication (exp2735–exp2736)
- exp2735: KV260 Continuity .259 (5th consecutive Branch B) (20 turns, codex)
- exp2736: Paper v6 arXiv Submission Package Prep (gated on exp2729.latex_compiles) (30 turns, codex)

### Phase F: Synthesis (exp2737)
- exp2737: Capstone v259 (80 turns, claude/opus, requires_claude: true)

---

## Dependency Graph

```
exp2725 (archive/activate)
    │
    ├── exp2726 (test collection fix)
    ├── exp2727 (verifier energy debug) ─────────── exp2731 (semantic energy tier 0g)
    ├── exp2728 (OTV fast path)
    ├── exp2729 (pdflatex + paper v6 theory v3) ── exp2736 (arXiv package prep)
    ├── exp2730 (HF mirror + ship v6 final)
    ├── exp2732 (entanglement retirement)
    ├── exp2733 (ORCA-NEXUS FR-11)
    ├── exp2734 (FALCON repair)
    └── exp2735 (KV260 .259)
              │
         exp2737 (capstone — reads all)
```

Note: exp2731 is listed as reading exp2727's diagnostic output to understand what format
GGUF responses are in, but it does NOT hard-gate on verifier_discriminative — Semantic Energy
works regardless of whether the old verifier was fixed.

---

## Hardware Requirements

- **RTX 3090 x2** (48 GB VRAM): exp2727 (debug run), exp2731 (Semantic Energy inference)
- **CPU only**: exp2725, exp2726, exp2728, exp2729, exp2730, exp2732, exp2733, exp2734, exp2736, exp2737
- **KV260**: exp2735 (SD card check — Branch B expected unless operator inserts card)

---

## Agent Routing

| Task | Agent | Model | Justification |
|------|-------|-------|---------------|
| exp2725–exp2736 (12 tasks) | codex | gpt-5.5 | Standard codex-default per CLAUDE.md |
| exp2737 (capstone) | claude | opus | requires_claude: true — 12+ file synthesis, open-ended judgment |

12/13 codex (92.3%) — within 2/13 ceiling. ✓

---

## CLAUDE.md Mandatory Discipline Checklist

- [x] **Codex-Default**: 12/13 codex (exp2737 is the only claude/opus — within ceiling)
- [x] **prior_failures**: exp2729 has prior_failures from exp2721 (blocked verdict). All 4 sub-fields present.
- [x] **PRECONDITIONS step 0**: All compute-bound tasks (exp2727, exp2728, exp2731, exp2733) include step 0
- [x] **Principle-annotated artifact fields**: Every REQUIRED ARTIFACT FIELD has a `principle:` annotation
- [x] **Terminal-prefix verdicts**: All prompts specify complete: / blocked_ honest_verdict prefixes
- [x] **FR-11 mandate**: exp2733 (ORCA-NEXUS Integration, continuous_self_learning_task: true)
- [x] **Hardware-Task Continuity**: exp2735 (KV260 NON-TERMINAL, 5th consecutive Branch B)
- [x] **Exclusion Manifest cross-check**: 0 scope matches against all retired experiment IDs
- [x] **Operator-Only publication**: exp2736 never submits to arXiv (produces package for operator)
- [x] **SOTA GGUF models**: exp2727 and exp2731 include Qwen3.6-35B-A3B-GGUF in MODEL_SPECS

---

## Expected Outcomes

| Task | Key Gate Field | Target |
|------|----------------|--------|
| exp2726 | collection_clean | true |
| exp2727 | verifier_discriminative | true (non-zero energy on 10+ examples) |
| exp2728 | otv_fast_path_viable | true (probe_auroc > 0.65) |
| exp2729 | latex_compiles | true |
| exp2730 | hf_model_card_updated | true |
| exp2731 | tier0g_auroc | >= 0.70 |
| exp2732 | entanglement_lineage_retired | true |
| exp2733 | orca_nexus_integration_viable | true |
| exp2734 | falcon_repair_integrated | true |
| exp2735 | branch_taken | B (5th consecutive) |
| exp2736 | submission_package_ready | true |
| exp2737 | n_criteria_met | >= 7/12 |
