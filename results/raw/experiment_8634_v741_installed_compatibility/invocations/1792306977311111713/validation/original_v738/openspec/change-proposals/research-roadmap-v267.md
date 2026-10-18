# Research Roadmap v267 — Multi-Corpus Evidence Harvest: FoVer Overfit Thesis + Delta H2 Fix

**Milestone:** 2026.05.267
**Author:** outer-loop (Claude) — 2026-05-21T15:57Z
**Prior milestone:** 2026.05.266 (1 of 12 tasks: capstone only; pre-test cascade blocked all research)
**Tasks:** exp2819–exp2829 (11 tasks)

---

## What Milestone .266 Proved (and Didn't)

Milestone .266 produced **one artifact** (exp2818 capstone) and **zero research artifacts**.
The root cause was a self-referential gate deadlock: the conductor's pre-test gate was failing
("1 failed, 80 passed in 2.14s"), and the only task that could fix it (exp2810) was itself gated
on the same pre-test that was failing. After 3 consecutive SKIPs, exp2810 was retired as
"merit-failure", cascade-blocking exp2811–2817 via `gated_on: exp2810`.

**Resolution (outer-loop, 2026-05-21):** The pre-test suite was run manually. Result: 81 passed.
The `.pretest-cache.json` now records "81 passed in 3.86s" (saved at 15:31:42Z). The conductor will
use this cached result as long as the Python-file fingerprint is unchanged. The self-referential
deadlock is broken.

This is the **4th consecutive zero-research-artifact milestone** (.263, .264, .265, .266 all
produced no experiments). The accumulated carry-forward is: exp2809 (archive), exp2811 (FoVer
leakage), exp2812 (MBPP), exp2813 (HumanEval), exp2814 (TruthfulQA), exp2815 (cross-corpus),
exp2816 (FUSE), exp2817 (paper §5 table).

**Key validated findings from .261 and earlier (still load-bearing):**

| Experiment | Finding | Status |
|-----------|---------|--------|
| exp2766 | Phase 4 FEP held_out_auroc=0.9989, fep_viable=true | Validated |
| exp2756 | Ensemble v12 k=18 integrated, +0.011 AUROC lift | Done |
| exp2755 | FR-11 Tier 4: cycle3 AUROC=0.9275, 34 NEXUS rules | Validated |
| exp2759 | Tier 0y ECE~5e-8 (differentiable conformal) | Done |
| exp2760 | Phase 1 v0.1.0b1 ship ready | Done (operator action pending) |
| exp2761 | Paper v6 theory v4: 28pp compiles | Done |
| exp2765 | Verifier live-GPU v4: energy_values=[0,0,0,0,0] on 5 examples | Open — zero-energy |
| exp2767/2791 | Delta H2 regression: 0/60 successes | Open — repair broken |

---

## Three Biggest Gaps

### Gap 1 (CRITICAL): Multi-Corpus Evaluation Still Zero — 4 Milestones Accumulated

**Impact:** The FoVer-overfit thesis (operator hypothesis: "could it be that the self-learning has
internalized the entire FoVer set?") remains entirely untested. The headline AUROC=0.9857 (exp2546)
is FoVer-only. Without MBPP, HumanEval, TruthfulQA data points, the paper-v6 §5 table cannot be
written and the "multi-corpus generalization" claim cannot be made. The operator directive
("we want to report both with and without learning") is also unanswered after 4 milestones.

**Resolution:** exp2820 (FoVer leakage isolation), exp2821 (MBPP), exp2822 (HumanEval),
exp2823 (TruthfulQA) — all carry-forwards from .266. None gated on any pre-test fix task
(the cascade is resolved at outer-loop level).

### Gap 2 (CRITICAL): Delta H2 Repair Pipeline Regression — 5+ Milestones

**Impact:** empirical_delta=0.000 (exp2754, .261). Verify-and-repair shows zero improvement over
baseline on FoVer violations. The repair pipeline is broken. This has been confirmed across
multiple milestones (.261 exp2754, .262 exp2767 attempts all rate-limited on gemini, .264 exp2791
planned as Claude Opus but never ran). Without a positive empirical_delta, paper-v6 §5 cannot claim
that repair adds value beyond detection.

**Resolution:** exp2826 (Delta H2 Regression Fix v3) — Claude Opus, requires_claude: true.
Prior_failures: exp2791 (planned opus, never ran), exp2767 (gemini rate-limited 3x), exp2754
(confirmed H2 regression). What's different: pre-test cascade resolved, Claude Opus explicitly
routed (bypasses gemini rate-limit root cause from .262).

### Gap 3 (HIGH): FUSE Zero-Labeled Weighting — Untested

**Impact:** The current ensemble v12 (k=18) was assembled with FoVer-labeled supervision. This
means the ensemble weights are optimized for FoVer. On other corpora, the weights may be suboptimal.
FUSE (arXiv:2604.18547) enables ensemble composition without any labeled data by using mutual-
agreement patterns across verifiers. FUSE-style weighting would give a fair cross-corpus comparison
that doesn't favor FoVer.

**Resolution:** exp2825 (FUSE Zero-Labeled Ensemble Weighting) — runs on FoVer validation split
to validate FUSE weights, then measures whether FUSE weights generalize better to other corpora.

---

## Architecture Snapshot (as of .266 end)

```
Carnot Verifier Ensemble (k=18 as of exp2756)
├── Tier 0 verifiers (fast): k=14 base verifiers
│   ├── Tier 0f: Weak-Strong policy (t_low < t_high — orientation FIXED exp2758)
│   ├── Tier 0v: Set-Consistency Energy Networks (exp2743)
│   ├── Tier 0w: Paraphrastic Consistency (exp2746)
│   ├── Tier 0x: Conformal Selective Acting (exp2757)
│   └── Tier 0y: Differentiable Conformal Training ECE~5e-8 (exp2759)
├── Tier 1 verifiers (medium): phase-3 and NLA-class
├── Phase 4 FEP Router: held_out_auroc=0.9989 (exp2766) ✓
├── FR-11 Tier 4 NEXUS: 34 rules, cycle3 AUROC=0.9275 (exp2755) ✓
└── Repair Pipeline: empirical_delta=0.000 (BROKEN — exp2754)

Key open issues:
  - Zero-energy degenerate on live GGUF outputs (exp2765: all [0,0,0,0,0])
  - Repair pipeline regression (delta H2)
  - Multi-corpus AUROC unmeasured (FoVer headline only)
  - FR-11 memory leakage hypothesis untested
```

---

## New Research Contributions (2026 sweep)

| Paper | ArXiv | Integration plan |
|-------|-------|-----------------|
| FUSE: Ensemble Composition Without Labels | 2604.18547 | exp2825 |
| EBM-CoT: Energy Calibration for Implicit CoT | 2511.07124 | exp2827 |
| Tool Receipts: Practical Hallucination Detection | 2603.10060 | paper-v6 §4 cite |
| V₁: Pairwise Self-Verification | 2603.04304 | paper-v6 §4 cite |
| Behavioral Entanglement Reweighting | 2604.07650 | exp2825 context |

---

## Phase Structure

### Phase A — Archive + Admin (exp2819)
Archive milestones .265 and .266 to research-complete.yaml (both were zero-artifact;
neither was archived in its own activation task). Activate .267.

### Phase B — Corpus Evaluations (exp2820–exp2823)
Four parallel corpus evaluations, all dual-condition (Condition A: full FR-11 state,
Condition B: FR-11 state reset). Answers the operator directive from 2026-05-21.

All Phase B tasks are independent; they can run in any order after exp2819.

- **exp2820**: FoVer Memory Leakage Isolation (continuous_self_learning, FR-11 mandate)
- **exp2821**: MBPP dual-condition (N=100, execution-based GT)
- **exp2822**: HumanEval dual-condition (N=164, execution-based GT)
- **exp2823**: TruthfulQA dual-condition (N=817 or subset)

### Phase C — Research Advancement (exp2824–exp2827)
- **exp2824**: Cross-Corpus Per-Verifier Matrix (gated on exp2820.leakage_delta measured)
- **exp2825**: FUSE zero-labeled weighting (independent)
- **exp2826**: Delta H2 regression fix (Claude Opus — independent, requires_claude: true)
- **exp2827**: EBM-CoT Tier 0ac verifier prototype (independent)

### Phase D — Publication (exp2828)
- **exp2828**: Paper v6 §5 dual-condition table (gated on exp2824.cross_corpus_matrix_built)

### Phase E — Capstone (exp2829)
- **exp2829**: Capstone v267 synthesis (Claude Opus, requires_claude: true)

---

## Dependency Graph

```
exp2819 (archive) ─── enables ──► exp2820 (FoVer leakage, FR-11)
                                  ├── exp2821 (MBPP)
                                  ├── exp2822 (HumanEval)
                                  └── exp2823 (TruthfulQA)

exp2820 ──► [if leakage_delta measured] ──► exp2824 (cross-corpus matrix)
exp2824 ──► [gated] ──► exp2828 (paper §5 table)

exp2825 (FUSE) ─── independent ───────────► exp2829 (capstone)
exp2826 (delta H2) ─── independent ──────► exp2829 (capstone)
exp2827 (EBM-CoT) ─── independent ───────► exp2829 (capstone)
```

---

## Hardware Requirements

All hardware boards are at terminal state. No mandatory hardware tasks.

| Board | Status | Action |
|-------|--------|--------|
| KV260 | TERMINAL (.260, 3.183μs latency) | None required |
| GateMate | TERMINAL (.247, bitstream flashed) | None required |
| PolarFire | TERMINAL (.241, dispatch validated) | None required |

---

## Agent Routing

| Task | Agent | Model | Justification |
|------|-------|-------|--------------|
| exp2819 archive | gemini | gemini-3.1-pro-preview | Admin task, no judgment needed |
| exp2820 FoVer leakage | gemini | gemini-3.1-pro-preview | Structured experiment, deterministic criteria |
| exp2821 MBPP | gemini | gemini-3.1-pro-preview | Corpus eval with clear execution GT |
| exp2822 HumanEval | gemini | gemini-3.1-pro-preview | Corpus eval with execution GT |
| exp2823 TruthfulQA | gemini | gemini-3.1-pro-preview | Corpus eval, structured scoring |
| exp2824 cross-corpus | gemini | gemini-3.1-pro-preview | Pattern synthesis from upstream results |
| exp2825 FUSE | gemini | gemini-3.1-pro-preview | Algorithm implementation, deterministic |
| exp2826 delta H2 | claude | opus | Multi-file git bisect + iterative hypothesis test |
| exp2827 EBM-CoT | gemini | gemini-3.1-pro-preview | New verifier implementation |
| exp2828 paper §5 | gemini | gemini-3.1-pro-preview | Table from upstream artifacts |
| exp2829 capstone | claude | opus | Cross-artifact synthesis judgment |

**Routing summary:** 9/11 gemini (81.8%) + 2/11 claude/opus (18.2%, within 2/11 ceiling)

---

## Acceptance Criteria

| # | Criterion | Gate |
|---|-----------|------|
| 1 | Archive .265 + .266 complete | exp2819.archived_milestones includes ["2026.05.265", "2026.05.266"] |
| 2 | FoVer leakage delta measured | exp2820.leakage_delta != null |
| 3 | MBPP dual-condition AUROC measured | exp2821.condition_a_production_auroc_mean != null |
| 4 | HumanEval dual-condition AUROC measured | exp2822.condition_a_production_auroc_mean != null |
| 5 | TruthfulQA dual-condition AUROC measured | exp2823.condition_a_production_auroc_mean != null |
| 6 | Cross-corpus matrix built | exp2824.cross_corpus_matrix_built = true |
| 7 | FUSE weighting evaluated | exp2825.fuse_vs_supervised_delta != null |
| 8 | Delta H2 root cause identified | exp2826.root_cause_identified = true OR blocked_* |
| 9 | EBM-CoT prototype AUROC measured | exp2827.ebm_cot_auroc != null |
| 10 | Paper §5 table generated | exp2828.latex_section_compiled = true |
| 11 | FR-11 mandate satisfied | exp2820.continuous_self_learning_task = true |
| 12 | Capstone written | exp2829.honest_verdict starts with complete: |

---

## CLAUDE.md Compliance Checklist

- [x] **Gemini-Default**: 9/11 gemini, 2/11 claude/opus (both requires_claude: true)
- [x] **prior_failures**: All carry-forward tasks (exp2820–exp2828) have full 4-field prior_failures
- [x] **PRECONDITIONS step 0**: All compute-bound tasks have PRECONDITIONS block
- [x] **Principle-annotated fields**: All REQUIRED ARTIFACT FIELDS carry principle: annotations
- [x] **Terminal-prefix verdicts**: All prompts specify honest_verdict MUST start with terminal prefix
- [x] **FR-11 mandate**: exp2820 (continuous_self_learning_task: true)
- [x] **Hardware continuity**: All 3 boards TERMINAL — no mandatory hardware tasks
- [x] **KV260 SSH-Not-SD-Card**: N/A (no KV260 tasks)
- [x] **Exclusion manifest cross-check**: 0 scope matches found
- [x] **Operator-only publication**: exp2828 generates table only, never submits
- [x] **Failed-experiment rerun discipline**: exp2826 prior_failures names exp2791/exp2767/exp2754
- [x] **No exp2810 gate**: cascade resolved at outer-loop level — no task gated on exp2810
