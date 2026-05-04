# Research Roadmap vNEXT — Milestone 2026.04.96

**Planned:** 2026-05-04
**Follows:** 2026.04.95 (Pre-Commit Safety + Paper Related Work + GRPO-VPS Training + Phase-5 Derisking A-C + Reward Gaming Defense)

## What Milestone .95 Proved

- **GRPO-VPS beats v4 (+10pp baseline)** — exp1220 confirmed step-level process supervision
  improves over the v4 structural warm-up baseline. VPS is now the established training floor.
- **Phase-5 in-situ prototype fully stable** — exp1222/1223 passed all 5 gates: energy decrease
  67%, no representation drift, no autocatalytic spiral, no null-space excavation, no
  catastrophic forgetting. Oracle accuracy held 1.0 across 1000 queries.
- **Spera Theorem 9.2 confirmed empirically** — exp1224 adversarial probe found pairwise
  conditional acceptance probability P(V_i|V_j)=1.000 across all k=3 pairs. Effective k
  collapsed from 3 to 1 (only V1/changes_grid provided independent signal). Root cause:
  snap_to_action quadrant-anchor decoder structurally guaranteed V0 and V2 for all inputs.
  Mandatory redesign before any scale-up.
- **Paper v6 related work complete** — exp1218 added 5 citations (EBT, LLaDA, Coconut, ODAR,
  Kona), updated thesis sentence, applied novelty-boundary discipline.
- **Boltzmann-GPT structural signal confirmed** — exp1226 AUROC=0.65 at random weights.
  Architecture is non-degenerate; contrastive CD training is the next step.
- **GRPO v6 FSPO+VPS wall-budget exhausted** — exp1221 ran 848s vs 480s budget;
  insufficient_logprob_coverage. Retry with wall_budget_s=1200 resolves this.
- **exp1225 (LLMs gaming verifiers) 3x max_turns:40 with codex** — codex insufficient.
  Retry with claude/opus max_turns:80 and STEP 0.
- **exp1217 (auto-populate prior_failures) DOOMED_RERUN_BLOCK** — the script task itself was
  missing prior_failures. Fixed in .96 with prior_failures populated pointing to exp1217.
- **exp1228 (milestone retro) 3x FAIL** — artifact_not_updated_past_bootstrap. Carry to .96.

## Three Biggest Gaps

1. **Verifier joint orthogonality audit is paper-blocking AND scale-up-blocking** (HIGHEST).
   exp1224 proved empirically that k_eff=1 in k=3 in-situ ensemble due to structural
   correlations. The k=6 production ensemble's novelty claim requires measuring the 6x6
   P(V_i|V_j) conditional acceptance matrix before paper-v6 submits. Mandatory per
   known-issues.md .96 pickup. Without this, paper-v6 fails reviewer scrutiny on the
   "unprecedented formal distinctness" claim.

2. **arXiv submission still pending** (CRITICAL). All 5 critical paper issues resolved (exp1205),
   related work done (exp1218), bundle v8 compiled (exp1206). Only blocker: orthogonality audit
   (exp1232). Gate exp1234 on matrix measured, then submit immediately.

3. **Phase-5-D intermediate scale not derisked** (production blocking). exp1222/1223 validated
   50K params / d=16. Three production-scale failure modes invisible at toy scale: mode collapse
   (1B+), MCMC mixing paralysis (d>=256), substrate shift (geometric phase transition). Mandatory
   .96/.97 pickup per known-issues.md. Intermediate validation at 100-300M params / d=128
   catches these cheaply (30-60 GPU-hours) before 1B+ scale-up.

## Architecture Diagram

```
Paper-v6 arXiv Target (exp1234, gated on exp1232)
    ↑
exp1232: 6x6 P(V_i|V_j) on k=6 production ensemble
    ↑
k=6 Production Verifier Ensemble:
  V1: Z3MathVerifier          (Tier 3, logic grounding)
  V2: ASTStructureVerifier     (Tier 2.5, structural)
  V3: SemanticConsistencyVerifier (Tier 2.7, causal)
  V4: ThinkPRMv2               (Tier 0a, pre-trained PRM)
  V5: SOSKANEnergyV3 4-bit     (Tier 1, energy AUROC=0.990)
  V6: SemEnergyProbe           (Tier 0c, fast logit-space)
  → AND-compose (k_eff=? after audit → exp1233 redesign)

GRPO Training Stack:
  v4 structural (+10pp) → VPS beats v4 (exp1220)
  → exp1235: GRPO v6 FSPO+VPS (1200s, DualGPU)
  → exp1236: execution-grounded credit (gated on v6 improvement)

Phase-5 In-Situ:
  5-A prototype (exp1222 done)
  5-B training loop (exp1223 done, 5/5 gates)
  5-C adversarial probe (exp1224 done, revision needed)
  5-D intermediate (exp1238: 100-300M, d=128, DualGPU)
  5-E production 1B+ (future .97/.98)

Phase-3 Substrate:
  NRGPT AUROC=0.921 baseline
  Boltzmann-GPT seed AUROC=0.65 (exp1226)
  → exp1237: contrastive CD training (target AUROC>0.80)
```

## Phase Descriptions

### Phase 0 — Infrastructure / Carry-Forwards (3 tasks, unconditional, MANDATORY)

Close out the .95 retro (exp1229 — exp1228 failed 3x), fix auto-populate prior_failures script
(exp1230 — exp1217 was itself missing prior_failures, a circular failure), and retry LLMs gaming
verifiers defense with claude/opus at full budget (exp1231 — codex hit max_turns:40 three times).

exp1229 runs first (closes .95). exp1230 and exp1231 run unconditionally afterward.

### Phase 1 — Verifier Joint Orthogonality Audit (MANDATORY, paper-blocking, 2 tasks)

exp1232 measures the 6x6 P(V_i|V_j) conditional acceptance matrix on the k=6 production
ensemble using the FoVer test corpus (held-out). Computes k_eff as count of structurally
independent verifiers (pairs with P<0.7). Generates a heatmap figure for paper-v6. For pairs
with P>0.7, proposes replacement candidates per the Spera Theorem 9.2 framework.

exp1233 implements the redesign: replaces identified correlated pairs with structurally
independent alternatives per exp1232's proposals. Reruns the 6x6 matrix to confirm improvement.
Target: k_eff >= 4, max pairwise P < 0.7.

### Phase 2 — arXiv Submission (CRITICAL, 1 task, gated on exp1232)

exp1234 integrates the orthogonality heatmap into paper-v6, recompiles the PDF, and submits
to arXiv. Gated on exp1232.pairwise_correlation_matrix_measured (not on k_eff threshold —
we submit with honest k_eff even if <6, per CLAUDE.md provenance rule).

### Phase 3 — GRPO Training Extension (2 tasks)

exp1235 retries GRPO v6 FSPO+VPS with wall_budget_s=1200 (2.5x vs exp1221's 480s). Adds
token regulation (arXiv 2511.00066) to address logprob coverage gap. DualGPU MANDATORY.

exp1236 applies execution-grounded credit assignment (arXiv 2603.16158) gated on exp1235
showing positive improvement over the VPS floor.

### Phase 4 — Boltzmann-GPT Contrastive Training (1 task)

exp1237 trains the Boltzmann-GPT layer with contrastive divergence on FoVer corpus. Seed
(exp1226) confirmed AUROC=0.65 at random weights — non-degenerate. Target AUROC > 0.80.
Uses theoretical energy gap formulation from arXiv 2512.18730 for the contrastive loss design.

### Phase 5 — Phase-5-D Intermediate Scale (1 task, DualGPU MANDATORY)

exp1238 implements and runs the intermediate-scale Phase-5-D prototype: 100-300M params,
d=128 latent, k=5 verifiers, 1000 queries on ARC-AGI-class distribution. Measures all 8
failure modes (5 toy-detectable + 3 production-scale-only: mode collapse entropy, MCMC mixing
autocorrelation, substrate shift L∞ saturation). Includes PPSEBM-style EBM replay buffer
(arXiv 2512.15658) as anti-forgetting defense. Acceptance: all 8 gates measured.

### Phase 6 — New Research (2 tasks)

exp1239 runs the NRGPT Frozen-Prefix evaluation (optional per known-issues.md): re-runs
the energy recurrence trace on the first token in isolation to classify the non-monotonicity
as (b) causal-context shift vs (c) learned non-conservative preconditioner. Needed for
paper-v6 §4 honest framing.

exp1240 ships the WOPR Kakuro puzzle cartridge: integer row/column sum constraints with
Ising energy E=Σ_row(sum_violation)^2 + Σ_col(sum_violation)^2, E=0 at valid solution.

### Phase 7 — Retro (1 task)

exp1241 evaluates all 13 success criteria. claude/opus, STEP 0, max_turns:100. Uses
AGENT_TYPE_RETRO=claude (not codex — codex retro failures in .93, .95 established this).

## Dependency Graph

```
exp1229 (retro-95)     unconditional, FIRST
exp1230 (autofill-v2)  unconditional, after exp1229
exp1231 (gaming-def)   unconditional, after exp1229

exp1232 (orth-audit)   unconditional
exp1233 (orth-redesign) gated: exp1232.pairwise_correlation_matrix_measured
exp1234 (arxiv-v6)     gated: exp1232.pairwise_correlation_matrix_measured

exp1235 (GRPO-v6)      unconditional (prior_failures:[exp1221])
exp1236 (exec-credit)  gated: exp1235.grpo_v6_improvement_pp > 0

exp1237 (bolt-gpt-cd)  unconditional (prior_failures:[exp1226])
exp1238 (phase5d)      unconditional
exp1239 (nrgpt-prefix) unconditional
exp1240 (kakuro)       unconditional

exp1241 (retro-96)     unconditional, LAST
```

## Hardware Requirements

- **exp1235**: DualGPU MANDATORY — both RTX 3090s, tensor_split=[0.5,0.5], wall_budget_s=1200
- **exp1237**: Single GPU (RTX 3090 CUDA), ~30 min CD training on FoVer corpus
- **exp1238**: DualGPU MANDATORY — 100-300M param encoder, distributed training
- **exp1231, exp1232, exp1233, exp1234**: CPU-only (analysis, paper edits, LaTeX)
- **exp1239, exp1240**: CPU-only

## arxiv Papers Added to research-references.md for .96

1. arXiv 2512.18730 — EBM theoretical lens for RL-tuned LMs (Boltzmann-GPT CD loss basis)
2. arXiv 2312.09244 — Reward model ensembles still correlate (HSIC decorrelation target)
3. arXiv 2601.01490 — Constraint distortion vs hallucination (GRPO v6 metric design)
4. arXiv 2603.16158 — Execution-grounded GRPO credit assignment (exp1236 basis)
5. arXiv 2511.00066 — Token-regulated GRPO stability (incorporated into exp1235)
6. arXiv 2512.15658 — PPSEBM continual learning (Phase-5-D replay buffer)
7. arXiv 2511.18689 — QuantKAN quantization framework (.97 edge deployment)

## Success Criteria (13)

1.  retro_95_complete (exp1229)
2.  autofill_script_v2_shipped (exp1230)
3.  gaming_defense_measured (exp1231)
4.  verifier_orthogonality_matrix_measured_6x6 (exp1232)
5.  k_eff_documented_and_honest (exp1232)
6.  verifier_redesign_k_eff_above_3 (exp1233)
7.  arxiv_v6_submitted (exp1234)
8.  grpo_v6_improvement_measured (exp1235)
9.  boltzmann_gpt_contrastive_auroc_above_0p80 (exp1237)
10. phase5d_all_8_gates_measured (exp1238)
11. nrgpt_frozen_prefix_resolved (exp1239)
12. kakuro_cartridge_shipped (exp1240)
13. retro_96_complete (exp1241)

## Estimated Wall Time

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| 0 Infrastructure | 3 | 60 min |
| 1 Orthogonality Audit | 2 | 80 min |
| 2 arXiv Submission | 1 | 35 min |
| 3 GRPO Training | 2 | 90 min |
| 4 Boltzmann-GPT | 1 | 40 min |
| 5 Phase-5-D | 1 | 70 min |
| 6 New Research | 2 | 35 min |
| 7 Retro | 1 | 25 min |
| **Total** | **13** | **~435 min** |
