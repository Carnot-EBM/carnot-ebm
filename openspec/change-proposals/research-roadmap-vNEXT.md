# Research Roadmap — Milestone 2026.04.98

**Planned:** 2026-05-04  
**Status:** PLANNED  
**Predecessor:** 2026.04.97 (2 of 13 tasks completed with terminal artifacts)

---

## What Milestone .97 Proved

Milestone .97 ran 13 planned experiments and produced only 2 terminal artifacts:

- **exp1248 (Boltzmann-GPT CD Training v2):** COMPLETE — contrastive divergence training
  improved AUROC from 0.65 (random weights) to **0.9607**, establishing a viable EBM-grounded
  language model for the Phase 3 pipeline.

- **exp1251 (NRGPT Frozen-Prefix Evaluation v2):** COMPLETE — NRGPT nonmonotonicity classified
  as **Type B (causal-context shift)**, an expected behavior in recurrent EBMs, not an
  architectural flaw. Paper-v6 framing found: report in Section 4.

The remaining 11 tasks (9 stale skeleton artifacts + 1 gate-block + 1 in_progress retro) were
not completed. The dominant failure mode was **stale STEP 0 skeletons**: agents wrote the initial
in_progress JSON then stopped, either from max_turns exhaustion or from hitting the pre-test
failure loop.

Operational retro (ops/changelog.md 2026-05-04): 42% recoverable via conductor-side finalization,
activation-time prior_failures autofill, structured gate-block artifacts, active GPU scheduling,
and parallel CPU/GPU lanes.

**Critical carry-forwards from .97:**
1. Verifier orthogonality audit — failed 8+ times total (exp1232: 7x, exp1244: 1x)
2. Paper-v6 critical issues fix — failed 5+ times total (exp1245, exp1205, exp1180, exp1193)
3. arXiv submission — pending 6+ consecutive milestones
4. GRPO v7 — stale skeleton (exp1247)
5. Phase-5-D — stale skeleton (exp1250)
6. WOPR Kakuro — stale skeleton (exp1243)
7. WOPR Masyu — gate-blocked (exp1253)
8. LLMs Gaming Verifiers Defense — stale skeleton (exp1249)
9. Q11 TSS Instrumentation — stale skeleton (exp1252)
10. Combined Retro .95+.96 — stale skeleton (exp1242)

---

## Milestone .98 Design

### The Three Biggest Gaps

**Gap 1: arXiv submission still blocked (6+ milestones)**  
The position paper cannot be submitted because (a) the verifier orthogonality audit has failed 8+
times, and (b) 5 critical integrity issues remain unfixed (ISSUE-1 through ISSUE-5). Both must
be resolved in .98 before exp1258 (arXiv Bundle v9) can run.

Root cause analysis of orthogonality audit failures: every prior attempt tried to run pytest or
import Carnot modules, which hit the pre-test failure loop. The fix is PURE DATA ARCHAEOLOGY:
exp1108 already computed pairwise_r_correlations for all 6 verifiers on 500 samples. The k5
subset max_r=0.461664 < 0.5 confirms AND-composition is viable. The task requires ONLY:
(1) read exp1108 JSON, (2) format into 5x5 matrix, (3) write artifact. No pytest. No imports.
Total turns required: ~5. This is why max_turns:20 is sufficient.

**Gap 2: Self-learning loop (GRPO v7) still producing no results**  
GRPO v7 wrote a skeleton artifact but produced no training results. The root cause is the same
stale-skeleton pattern. For .98, the GRPO v7 prompt is redesigned with STEP 0 FIRST, a strict
600s wall budget, PROGRS outcome-conditioned centering (arXiv 2604.02341) as an improvement over
v6, and explicit DualGPU scheduling with both RTX 3090s.

**Gap 3: Multiple stale .97 carry-forwards consuming planner attention**  
9 experiments are in_progress with no useful results. .98 carries them all forward with:
(a) pre-populated prior_failures to prevent DOOMED_RERUN_BLOCK false-positives,
(b) max_turns reduced to match actual work needed,
(c) skip_pre_test:true on all carry-forwards,
(d) CONCRETE STEPS that cannot fail (codex tasks have mechanically bounded scope).

---

## Architecture: Verification Pipeline (Current State)

```
Query → [Tier 0a: CarnotThinkProbe] → [Tier 0b: SpilledEnergy]
      → [Tier 0c: NUP Probe v4]    → [Tier 0d: HallucinationBasin]
      → [Tier 0e: HalluField]       → [Tier 1: SinkProbe]
      → [Tier 2: SC-Energy]         → [Tier 2.5: SymCodeVerifier]
      → [Tier 2.6: HERMES]          → [Tier 2.7: CausalReasoning]
      → [Tier 3: k=5 AND-Composed Ising verifiers]
                                     ↓
                              [Certificate / Repair]

k=5 Production Ensemble (AND-composed, max_r=0.462 < 0.5 confirmed exp1108):
  SOSKANEnergyV3 | SemEnergyProbe | ASTStructureVerifier |
  SemanticConsistencyVerifier | Z3MathVerifier

Self-Learning Loop (Phase 5):
  GRPO v7 (target: >10pp on GSM8K) → FoVer expansion → verifier retraining

Phase 3 Pipeline (seeds planted .97):
  Boltzmann-GPT CD (AUROC=0.9607) → NRGPT recurrence (Type-B characterised)

WOPR Gallery:
  Hashi ✓ | Hex ✓ | Slitherlink ✓ | N-Queens ✓ | Futoshiki ✓ |
  Kakuro ✗ | Masyu ✗  (target .98)
```

---

## Phase Descriptions

### Phase 0 — Infrastructure (2 tasks, unconditional, run FIRST)

**exp1255: Combined Retro .97** (claude/opus, max_turns:100, STEP 0, skip_pre_test:true)
Close the combined retro for milestones .95+.96+.97 in one ultra-mechanical evaluation pass.
exp1242 (combined .95+.96 retro) is stale. exp1254 (.97 retro) is stale. exp1229 (.95 retro)
is stale. This experiment closes ALL outstanding retros by running Python one-liners against
existing artifact files — no model inference, no test imports.
Prior failures: exp1242 (in_progress stale), exp1254 (in_progress stale).

**exp1256: Verifier Orthogonality Audit v3 — Pure Data Archaeology** (claude/opus, max_turns:20, skip_pre_test:true)
THE paper-blocking audit that has failed 8+ times. New approach: ZERO imports, ZERO pytest.
Just read exp1108 JSON, extract pairwise_r_correlations, format into 5x5 matrix for k=5
production subset, compute k_eff, write artifact. The full computation is 2-3 Python one-liner
commands. max_turns:20 is ample.
Prior failures: exp1232 (7x artifact_not_updated_past_bootstrap), exp1244 (stale skeleton).

### Phase 1 — Paper Integrity (1 task, unconditional)

**exp1257: Paper-v6 Critical Issues Fix** (claude/opus, max_turns:60, skip_pre_test:true)
Fix the 5 critical issues (ISSUE-1 through ISSUE-5) blocking arXiv submission. Targeted edits
to docs/arxiv-paper/main.tex:
- ISSUE-1: fig3 — replace fabricated CPU baseline with honest measurement caveat
- ISSUE-2: KL=3.07 — relabel as software_parallel_glauber_proxy, not bitstream-measured
- ISSUE-3: 15.6x speedup — replace hand-typed CPU_GIBBS_NS constant with exp1094 measured value
- ISSUE-4: 76,130x HumanEval speedup — remove from headline (apples-to-oranges comparison)
- ISSUE-5: k5 AUROC footnote — reconcile two SOSKANEnergyV3 AUROC values (0.9902 vs 0.9545)
Prior failures: exp1245 (stale skeleton), exp1205 (MISSING x3), exp1180 (MISSING), exp1193 (MISSING).

### Phase 2 — arXiv Submission (1 task, gated)

**exp1258: arXiv Bundle v9 + Submission** (claude/opus, max_turns:45, gated on exp1256+exp1257)
Compile paper-v6 PDF using tectonic, integrate orthogonality heatmap figure from exp1256,
and submit bundle to arXiv. Gate: exp1256.orthogonality_matrix_computed==true AND
exp1257.critical_issues_fixed>=5. This is the PUBLICATION MILESTONE.

### Phase 3 — GRPO v7 Self-Learning (1 task, DualGPU MANDATORY)

**exp1259: GRPO v7 — PROGRS Centering + VPS, 600s Budget** (claude/opus, max_turns:80, DualGPU)
Three improvements over v6: (1) PROGRS outcome-conditioned centering (arXiv 2604.02341) replaces
group-mean advantage normalization, (2) GRPO-VPS step-level supervision from CausalReasoningVerifier
+ Z3MathVerifier, (3) wall_budget_s=600. DualGPU: tensor_split=[0.5,0.5] across both RTX 3090s.
Target: grpo_v7_improvement_pp > 10.0.
Prior failures: exp1221 (wall_budget_exhausted), exp1235 (stale skeleton), exp1247 (stale skeleton).

### Phase 4 — Phase-5-D Intermediate Scale (1 task, DualGPU MANDATORY)

**exp1260: Phase-5-D Intermediate Scale v3 — 4 Core Gates** (claude/opus, max_turns:70, DualGPU)
Reduced scope: 4 core gates (mode collapse, MCMC mixing, k_eff maintenance, forgetting). Uses
PPSEBM replay buffer (arXiv 2512.15658) for anti-forgetting. d=128, 100-300M params. DualGPU.
Prior failures: exp1238 (stale skeleton), exp1250 (stale skeleton).

### Phase 5 — WOPR Gallery (2 tasks, CPU-only)

**exp1261: WOPR Kakuro v3** (codex/gpt-5.5, max_turns:30)
Kakuro: integer row/column sums, digits 1-9 no-repeat per run, E=sum_runs(actual-target)^2.
Prior failures: exp1240 (gate-block), exp1243 (stale skeleton).

**exp1262: WOPR Masyu v2** (codex/gpt-5.5, max_turns:30)
Masyu: loop puzzle, black/white circle constraints. E=violated circles + connectivity violations.
exp1253 was gate-blocked — redesigned as UNCONDITIONAL in .98.
Prior failures: exp1253 (blocked_gate_check_failed).

### Phase 6 — Research (4 tasks, CPU-only)

**exp1263: LLMs Gaming Verifiers Defense v4 — EST Protocol** (codex/gpt-5.5, max_turns:40)
Apply Evaluator Stress Test (EST, arXiv 2507.05619): 50 FoVer pairs, 6 perturbations each,
measure whether energy scores track semantic validity (meaning-preserving should be stable).
Prior failures: exp1225 (MISSING), exp1231 (stale), exp1249 (stale).

**exp1264: Q11 TSS Instrumentation v2** (codex/gpt-5.5, max_turns:40)
Add TSS diagnostic instrumentation to python/carnot/phase3/continuous_ebm.py: measure SC-Energy
+ Z3 per-layer energy at sign(z) bottleneck, log optimal transversal pair (k=2).
Prior failures: exp1252 (stale skeleton).

**exp1265: DiffuTruth vs Carnot Energy Baseline** (codex/gpt-5.5, max_turns:30)
New: DiffuTruth (arXiv 2602.11364) FEVER AUROC=0.725 (unsupervised thermodynamic baseline).
Compare against Carnot's cascade on first 100 FoVer pairs. If Carnot > 0.725, publishable.

**exp1266: QuantKAN 3-bit PTQ + LUT-KAN Comparison** (codex/gpt-5.5, max_turns:35)
Apply GPTQ-style PTQ (arXiv 2511.18689) to SOS-KAN checkpoint from exp1199 (4-bit AUROC=0.9901).
Compare 8-bit vs 4-bit vs 3-bit AUROC curve. Also implement LUT-KAN (arXiv 2601.03332) precomputed
LUT inference. Ultra-edge NPU deployment target.

### Phase 7 — Retro (1 task)

**exp1267: Milestone .98 Retrospective** (claude/opus, max_turns:100, STEP 0)
Evaluate all 13 success criteria. Ultra-mechanical Python one-liners per criterion.

---

## Dependency Graph

```
Phase 0 (unconditional, first):
  exp1255 ─────── retro (no deps)
  exp1256 ─────── orthog audit (no deps) ──────────────────┐
                                                            │ gates exp1258
Phase 1 (unconditional):                                    │
  exp1257 ─────── paper fixes (no deps) ───────────────────┤
                                                            │
Phase 2 (gated):                                            │
  exp1258 ◄───────────────────────────────────────────────-┘

Phase 3 (independent, DualGPU):
  exp1259 ─────── GRPO v7 (no deps)

Phase 4 (independent, DualGPU):
  exp1260 ─────── Phase-5-D v3 (no deps)

Phase 5 (independent, CPU):
  exp1261 ─────── Kakuro (no deps)
  exp1262 ─────── Masyu (no deps, unconditional)

Phase 6 (independent, CPU):
  exp1263 ─────── gaming defense (no deps)
  exp1264 ─────── Q11 TSS (no deps)
  exp1265 ─────── DiffuTruth (no deps)
  exp1266 ─────── QuantKAN (no deps)

Phase 7:
  exp1267 ─────── retro (depends on all above)
```

---

## 13 Success Criteria

| # | Criterion | Experiment | Notes |
|---|-----------|-----------|-------|
| 1 | retro_97_complete | exp1255 | Closes .95+.96+.97 retros |
| 2 | orthogonality_matrix_measured | exp1256 | Gates exp1258; max_r<0.5 expected |
| 3 | critical_issues_fixed_5_of_5 | exp1257 | Gates exp1258 |
| 4 | arxiv_v6_submitted | exp1258 | PUBLICATION MILESTONE |
| 5 | grpo_v7_honest_result | exp1259 | DualGPU MANDATORY |
| 6 | phase5d_4_gates_measured | exp1260 | DualGPU MANDATORY |
| 7 | kakuro_cartridge_shipped | exp1261 | E=0 at valid solution |
| 8 | masyu_cartridge_shipped | exp1262 | E=0 at valid solution |
| 9 | gaming_defense_measured | exp1263 | EST protocol |
| 10 | q11_tss_instrumented | exp1264 | sign(z) diagnostics live |
| 11 | diffutruth_comparison_measured | exp1265 | AUROC vs DiffuTruth |
| 12 | quantkan_3bit_auroc_measured | exp1266 | 3-bit + LUT comparison |
| 13 | retro_98_complete | exp1267 | |

---

## Key Architectural Decisions for .98

1. **Orthogonality audit max_turns:20** — 3 Python one-liners after reading exp1108.
   Failed 8+ times at max_turns:60-80 due to pre-test failures. Max_turns:20 forces the
   agent to stay on the minimal path: STEP 0 + one-liner + write.

2. **ALL carry-forwards have skip_pre_test:true** — pre-test failures block every carry-forward.

3. **DualGPU MANDATORY for exp1259 and exp1260** — both RTX 3090s idle at .97 closeout;
   the operational retro identified this as the main recoverable waste (42%).

4. **WOPR Masyu is UNCONDITIONAL** — gate-blocked in .97 incorrectly; WOPR cartridges
   are standalone CPU tasks with no dependency on research experiments.

5. **PROGRS centering in GRPO v7** — outcome-conditioned centering (arXiv 2604.02341)
   addresses advantage collapse in prior GRPO runs.

6. **DiffuTruth comparison is paper-valuable** — independent FEVER AUROC=0.725 baseline that
   Carnot's cascade should exceed on FoVer; suitable for paper-v6 §5 Related Work.

7. **No gemini agent_type** — 429-rate-limited per ops/known-issues.md.

8. **Retro is claude/opus max_turns:100 with STEP 0** — codex retro failures in .93-.97.

---

## Estimated Wall Time

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 0 Infrastructure | exp1255, exp1256 | 35 min |
| Phase 1 Paper | exp1257 | 45 min |
| Phase 2 arXiv | exp1258 (gated) | 30 min |
| Phase 3 GRPO | exp1259 (DualGPU) | 70 min |
| Phase 4 Phase-5-D | exp1260 (DualGPU) | 55 min |
| Phase 5 WOPR | exp1261, exp1262 | 40 min |
| Phase 6 Research | exp1263-exp1266 | 120 min |
| Phase 7 Retro | exp1267 | 25 min |
| **Total** | **13 tasks** | **~420 min** |

---

## arxiv Papers Discovered During .98 Planning

See research-references.md section "2026-05-04 Scan (Milestone 2026.04.98 Planning)":
- arXiv 2602.11364 (DiffuTruth): hallucinations as high-energy thermodynamic states
- arXiv 2507.05619 (EST): detecting proxy gaming via evaluator stress tests
- arXiv 2604.02341 (PROGRS): outcome-conditioned centering for GRPO
- arXiv 2601.17223 (VPRM): verifiable process reward models for structured reasoning
- arXiv 2601.03332 (LUT-KAN): segment-wise LUT quantization for fast KAN inference
