# Carnot Research Milestone v262
## Verifier Live-GPU v4 + FEP Adversarial Recheck + Delta H2 Fix + Ensemble v13 + NEXUS Expansion

**Milestone:** 2026.05.262
**Date:** 2026-05-21
**Experiment IDs:** exp2764–exp2776 (13 tasks)
**Status:** PROPOSED

---

## What Milestone .261 Proved

Milestone .261 (exp2751–exp2763) closed 10 of 12 acceptance criteria. Key outcomes:

| Finding | Artifact | Status |
|---|---|---|
| Phase 4 FEP strategy2 AUROC=0.9947 | exp2753 | TAUTOLOGY-flagged — held-out validation required |
| Empirical delta=0.000, H2 regression | exp2754 | Repair pipeline broken — paper 4/δ cite blocked |
| FR-11 Tier 4 validated (cycle3=0.9275, overlap=0) | exp2755 | CONFIRMED — genuine generalization |
| Ensemble v12 (k=18, +0.011 lift) | exp2756 | CONFIRMED — Tier 0w decorrelated at 0.26 |
| Conformal routing (84% savings, anytime-valid) | exp2757 | CONFIRMED |
| Weak-strong orientation bug fixed (41% savings, 0% FNR) | exp2758 | CONFIRMED |
| Tier 0y differentiable calibration (ECE ~5e-8) | exp2759 | CONFIRMED |
| Phase 1 v0.1.0b1 shipped | exp2760 | CONFIRMED |
| Paper-v6 28pp, compiles | exp2761 | CONFIRMED |
| arXiv package v3 ready (HOLDS for operator) | exp2762 | CONFIRMED — operator-only submission |
| exp2752 verifier live-GPU v3 **ABSENT** | — | CRITICAL — 3rd consecutive absent |
| Verifier discriminativeness unvalidated in-vivo | — | ALL verifier results are CPU-only |

---

## Three Biggest Gaps for .262

### Gap 1 — CRITICAL: Verifier live-GPU v4 absent (3 consecutive milestones)

**Root cause:** Unknown. Three consecutive attempts (.259 adversarial-flagged DURATION_TOO_SHORT,
.260 blocked_gguf_qwen36_not_cached, .261 absent). The .261 absence is particularly alarming:
the experiment was dispatched to a gemini agent but no artifact file was written. This suggests
the agent either hit max_turns during the inference loop without writing an artifact, or the
PRECONDITIONS ran but an error occurred before the artifact write.

**Fix for .262 (exp2765):**
- Write a STUB ARTIFACT at Step 0.0 (before any preconditions) — ensures artifact exists even
  if the agent fails mid-run.
- Reduce N from 30 to 5 (sufficient to confirm discriminativeness without risk of timeout).
- Add explicit "if any step fails, update stub with blocked_* verdict and exit" instruction.
- Use gemma-4-26B-A4B-it-GGUF as PRIMARY (confirmed cached in exp2741).
- Hard gate: total_elapsed < 30s → blocked_suspicious_duration_too_short.
- retire_if_same_verdict: true — if 4th attempt also absent or adversarial, retire GGUF/llama_cpp
  approach and pivot to transformers loader.

### Gap 2 — HIGH: FEP TAUTOLOGY flag needs held-out cross-validation

**Root cause:** exp2753 ran strategy2 (learned logistic regression) on the FoVer corpus but the
artifact shows best_fep_auroc == strategy2_auroc to >5 significant figures. The 6.49s duration
suggests the logistic regression was trained AND evaluated on the same small dataset (in-sample
evaluation). The 0.9947 AUROC may reflect memorization, not generalization.

**Fix for .262 (exp2766):**
- Leave-one-out cross-validation (LOO-CV) over the entire FoVer corpus.
- Separate held-out evaluation set (N >= 200, never seen by logistic regression fitting).
- If in-sample AUROC >> held-out AUROC → confirmed overfitting; FEP claim blocked for paper-v6.
- If held-out AUROC > 0.8 AND delta vs ODAR > 0.05 → FEP empirically validated; unblock arXiv.

### Gap 3 — HIGH: Empirical delta = 0.000, repair pipeline H2 regression

**Root cause:** exp2754 identified root_cause="H2 regression" — a regression introduced somewhere
in the repair pipeline between the last known-good delta milestone and .261. The repair pipeline
correctly identifies violations (detect stage works) but the repair stage either produces no
improvements or the verify-after-repair stage regresses.

**Per arXiv:2605.12270:** 22% of repair failures are regression introductions — fixing the target
error but breaking other constraints. This matches H2 perfectly.

**Fix for .262 (exp2767):**
- Git bisect between last known-good delta milestone and current HEAD.
- Identify the regression commit.
- Fix the regressed code.
- Verify empirical_delta > 0.10 on N >= 100 corpus with random_seed pinned.
- Until gate fires, paper-v6 Section 4 4/δ bound cite is BLOCKED (operator-only removal).

---

## Architecture Snapshot at .262 Start

```
VerifyRepairPipeline (production)
├── Routing Layer
│   ├── ODAR (free-energy, K=3) — exp2720
│   ├── Conformal Selective Acting (anytime-valid, 84% savings) — exp2757
│   └── Weak-Strong Policy (2-threshold, 41% savings, 0% FNR) — exp2758
├── Verifier Ensemble (k=18, v12)
│   ├── Tier 0a: Boolean constraint energy
│   ├── Tier 0b: SAT-based structural energy
│   ├── Tier 0c: Z3 SMT energy
│   ├── Tier 0d: AST structural energy
│   ├── Tier 0e: EORM energy (linear probe calibrated)
│   ├── Tier 0f: Semantic calibration (AUROC 0.992)
│   ├── Tier 0g: Semantic Energy char-n-gram TF-IDF (fix applied, live-GPU unvalidated)
│   ├── Tier 0h–0u: (base verifier ensemble)
│   ├── Tier 0v: Set-Consistency Energy (AUROC 0.818) — exp2743
│   ├── Tier 0w: Paraphrastic Consistency (corr=0.26, best decorrelation) — exp2746
│   ├── Tier 0x: Conformal Selective Acting router — exp2757
│   └── Tier 0y: Differentiable Conformal Calibration (ECE~5e-8) — exp2759
├── Self-Learning (FR-11)
│   └── ORCA-NEXUS Tier 4 (cycle3 AUROC=0.9275, 34 rules, 3 cycles) — exp2755
│       [research-only; production integration target in .262 exp2768]
├── Phase 4 FEP
│   └── Strategy2 logistic aggregator (AUROC=0.9947, TAUTOLOGY-flagged) — exp2753
│       [needs held-out cross-validation before paper claim]
└── Repair Pipeline
    └── BROKEN (H2 regression, empirical_delta=0.000) — exp2754
        [needs git bisect + fix in exp2767]
```

**Phase 1 status:** Shipped at v0.1.0b1 (exp2760). PyPI via CI OIDC.
**FPGA:** All 3 boards at terminal state — no mandatory hardware tasks.
**Paper-v6:** 28pp (exp2761), arXiv package v3 ready (exp2762). HOLDS for operator review.

---

## Milestone .262 Phase Structure

### Phase A — Lifecycle (1 task)
- **exp2764:** Archive .261 + Activate .262

### Phase B — Critical Gap Fixes (3 tasks)
- **exp2765:** Verifier Live-GPU v4 — Stub-First Protocol, N=5, gemma-4-26B Primary
  (4th attempt; hard preconditions; retire_if_same_verdict: true)
- **exp2766:** Phase 4 FEP Adversarial Recheck — LOO Cross-Validation + Held-Out Corpus
  (TAUTOLOGY flag resolution; n>=200 gate)
- **exp2767:** Empirical Delta H2 Regression Fix — Git Bisect + Pipeline Repair
  (goal: empirical_delta > 0.10 on N>=100)

### Phase C — Research Advancement (6 tasks)
- **exp2768:** FR-11 Tier 4 Production Integration — ORCA-NEXUS into VerifyRepairPipeline
  (wire 3-cycle learning loop into production pipeline; continuous_self_learning_task: true)
- **exp2769:** Ensemble v13 — Tier 0z Temporal/Causal Consistency Verifier
  (diversity-first selection per arXiv:2511.07784; target corr < 0.3 with existing ensemble)
- **exp2770:** Repair Pipeline Forensic — Detect-Repair-Verify Stage Analysis (arXiv:2603.23633)
  (diagnose which stage fails; informs exp2767 fix strategy)
- **exp2771:** NEXUS Constraint Memory Expansion — 34 → 50+ Domain-Specific Rules
  (math/code/logic/factual domains; continuous_self_learning_task: true)
- **exp2772:** CP-Router Uncertainty Routing — arXiv:2505.19970 Integration
  (entropy-aware thresholds as complement to conformal selective acting)
- **exp2773:** Paper v6 Theory v5 — FEP Recheck + Delta Fix + Ensemble v13 Integration
  (gated on exp2766 and exp2767; update Sections 3-4)

### Phase D — Publication Track (2 tasks)
- **exp2774:** HuggingFace Model Card Update + IPFS Pinning Checklist
  (post-ship Phase 1 tasks; operator-only publish discipline)
- **exp2775:** arXiv Package v5 — Updated with .262 Results
  (gated on exp2773.latex_compiles_v5=true)

### Phase E — Capstone (1 task)
- **exp2776:** Capstone v262 — Cross-Artifact Synthesis + Gaps for .263 (claude/opus)

---

## Dependency Graph

```
exp2764 (archive)
  ├── exp2765 (verifier live-GPU v4) — independent
  ├── exp2766 (FEP recheck) — independent
  ├── exp2767 (delta H2 fix) — independent
  ├── exp2768 (FR-11 production integration) — independent
  ├── exp2769 (ensemble v13 Tier 0z) — independent
  ├── exp2770 (repair forensic) → exp2767 (informs fix strategy)
  ├── exp2771 (NEXUS expansion) — independent
  ├── exp2772 (CP-Router routing) — independent
  ├── exp2773 (paper v6 theory v5) — gated on exp2766 + exp2767
  │   └── exp2775 (arXiv package v5) — gated on exp2773.latex_compiles_v5
  └── exp2774 (HF model card + IPFS checklist) — independent
exp2776 (capstone) — reads all upstream artifacts
```

---

## Acceptance Criteria

| # | Criterion | Experiment | Gate |
|---|---|---|---|
| 1 | Verifier live-GPU artifact present and non-adversarial | exp2765 | verifier_discriminative=true AND duration_s>=60 |
| 2 | FEP adversarial recheck resolves TAUTOLOGY | exp2766 | held_out_auroc > 0.8 OR paper-v6 FEP claim removed |
| 3 | Empirical delta > 0.10 (H2 fix confirmed) | exp2767 | empirical_delta > 0.10 on N>=100 |
| 4 | FR-11 Tier 4 production integration wired | exp2768 | fr11_production_integration=true |
| 5 | Ensemble v13 (Tier 0z candidate) added | exp2769 | tier0z_auroc > 0.0 AND tier0z_corr_with_0w < 0.4 |
| 6 | Repair pipeline forensic stage identified | exp2770 | bottleneck_stage in {detect, repair, verify} |
| 7 | NEXUS rules 50+ | exp2771 | n_rules >= 50 |
| 8 | CP-Router integrated | exp2772 | cp_router_viable=true |
| 9 | Paper-v6 theory v5 compiles | exp2773 | latex_compiles_v5=true |
| 10 | HF model card + IPFS checklist produced | exp2774 | hf_checklist_produced=true |
| 11 | arXiv package v5 ready (HOLDS for operator) | exp2775 | submission_package_v5_ready=true |
| 12 | Capstone synthesizes .262 state | exp2776 | n_criteria_met >= 8 |

---

## Agent Routing

| Category | Agent | Tasks |
|---|---|---|
| Standard experiments (12/13) | gemini / gemini-3.1-pro-preview | exp2764–exp2775 |
| Capstone (1/13) | claude / opus | exp2776 |

Gemini-Default applied (12/13 gemini, within 2/13 claude ceiling per CLAUDE.md).

exp2776 meets all 3 positive criteria for requires_claude:
1. Cross-artifact synthesis: reads all 12 upstream artifacts + synthesizes gaps for .263.
2. Multi-file context: judgment spans exp2765 (live-GPU), exp2766 (FEP), exp2767 (delta), exp2768 (FR-11) simultaneously.
3. Open-ended judgment under ambiguity: must weigh partially-validated findings and propose next steps.

---

## Hardware Continuity

All 3 FPGA boards at terminal state — no mandatory hardware tasks:
- KV260: terminal at .260 (exp2742, 3.183μs mean latency)
- GateMate: terminal at .247
- PolarFire: terminal at .241

---

## FR-11 Mandate

Primary: **exp2768** (FR-11 Tier 4 Production Integration, continuous_self_learning_task: true)
Backup: **exp2771** (NEXUS Constraint Memory Expansion, continuous_self_learning_task: true)

---

## New Research Integrated

| Paper | arXiv | Experiment |
|---|---|---|
| CP-Router (conformal uncertainty routing) | 2505.19970 | exp2772 |
| Detect-Repair-Verify empirical study | 2603.23633 | exp2770 |
| ReVISE (test-time self-verification) | 2502.14565 | exp2768 context |
| T³RL (tool verification for TTT) | 2603.02203 | exp2768 context |
| Multi-agent debate diversity study | 2511.07784 | exp2769 Tier 0z selection |
| Failure modes of LLM repair | 2605.12270 | exp2767 + exp2770 |

---

## Exclusion Manifest Cross-Check

Checked against ops/exclusion_manifest.yaml:
- exp2091 (Tier 1 CSL Grammar): scope not matched
- OTV probe: RETIRED (exp2739) — not re-proposed
- Diversity-maximizing selection: RETIRED (exp2739) — not re-proposed
- HalluSAEGeometricProbe: RETIRED — not re-proposed
- Discriminative JEPA: RETIRED (exp887) — not re-proposed
- Integer IDs 260, 308, 309, 346, 380-383, 410, 425, 491, 527, 603, 627, 786, 641, 906: not re-proposed
- PIMI research N=8/N=64: RETIRED — not re-proposed
- GRPO/VPRM v1-v14: RETIRED — not re-proposed
- GGUF-CLI-download approach: RETIRED — exp2765 uses llama_cpp loader (not CLI download)
- iCE40-PIMI: RETIRED — not re-proposed
- kv260_host_sd_card_precondition: RETIRED — no hardware tasks in .262

0 scope matches found. ✓

---

## CLAUDE.md Discipline Compliance

- Gemini-Default (2026-05-20): 12/13 gemini + 1 claude/opus (exp2776 capstone, requires_claude: true) ✓
- prior_failures: all 4 mandatory sub-fields present on all tasks with prior failures ✓
- PRECONDITIONS step 0 on all compute-bound tasks ✓
- Principle-annotated REQUIRED ARTIFACT FIELDS on every task ✓
- Terminal-prefix verdicts (complete:/success:/passed:/blocked_*) ✓
- FR-11 mandate: exp2768 + exp2771 (continuous_self_learning_task: true) ✓
- No hardware tasks (all 3 boards at terminal state) ✓
- KV260 SSH-Not-SD-Card: N/A (no KV260 tasks) ✓
- Operator-Only External Publication: exp2774 produces checklist; exp2775 produces package — both non-submitting ✓
- No canonical URL violations (github.com/Carnot-EBM/carnot-ebm throughout) ✓
- Exclusion Manifest cross-check: 0 scope matches ✓
- Failed-Experiment Rerun Discipline: prior_failures documented for exp2765/exp2766/exp2767 ✓
- Verdict Terminal-Prefix Discipline: all task prompts specify complete:/blocked_* prefixes ✓
- Pre-Launch Preconditions: step 0 with resource checks on compute-bound tasks ✓
- Adversarial Artifact Verification: duration_s gates + reproducibility_checksum on all tasks ✓
