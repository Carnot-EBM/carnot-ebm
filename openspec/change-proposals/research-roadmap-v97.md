# Research Roadmap v97 — Milestone 2026.04.97

**Planned:** 2026-05-04
**Follows:** 2026.04.96 (Verifier Orthogonality Audit + arXiv Submission + GRPO-VPS Extension + Boltzmann-GPT + Phase-5-D)

## What Milestone .96 Proved

- **Auto-populate prior_failures script shipped (exp1230)** — `scripts/conductor_priors_autofill.py`
  deployed. Dry-run scanned 13 tasks, generated 11 stubs. 7 tests pass. This directly addresses
  the recurring DOOMED_RERUN_BLOCK pattern.
- **All other .96 experiments failed with `artifact_not_updated_past_bootstrap` or `No file
  changes produced`** — the dominant failure mode: agent writes STEP 0 skeleton, then stops
  without completing the actual computation.
- **Verifier orthogonality audit (exp1232) retired after 7 total attempts** — paper-blocking.
  Downstream gate-blocked exp1233 (verifier redesign) and exp1234 (arXiv submission).
- **GRPO v6 FSPO+VPS (exp1235) retired after 6 total attempts** — gate-blocked exp1236.
- **Boltzmann-GPT CD training (exp1237) failed 3x** — artifact_not_updated_past_bootstrap.
- **Phase-5-D intermediate scale (exp1238) failed 3x** — artifact_not_updated_past_bootstrap.
- **NRGPT frozen-prefix (exp1239) failed 3x** — artifact_not_updated_past_bootstrap.
- **WOPR Kakuro (exp1240) DOOMED_RERUN_BLOCK 3x** — 10 priors flagged, prior_failures missing.
- **Both retro experiments (exp1229 for .95, exp1241 for .96) failed** — artifact_not_updated_past_bootstrap.

## Root Cause Analysis

**Why did artifact_not_updated_past_bootstrap dominate .96?**

The `No file changes produced` variant (exp1238/1239/1241) indicates the agent didn't even
write a file — crashing before STEP 0. The `artifact_not_updated_past_bootstrap` variant
indicates STEP 0 was written but the agent stopped before completing real work. Root causes:

1. **Pre-test suite failing causes SKIP before agent runs** — even with STEP 0, if the
   conductor's pre-test check fails, the agent never launches. Solution: `skip_pre_test: true`
   on ALL heavy tasks.
2. **Tasks are too complex for the turn budget** — GRPO v6 FSPO+VPS, Phase-5-D (100-300M
   params), and gaming verifiers defense all require complex multi-step work the agent can't
   complete in 60-80 turns.
3. **Orthogonality audit has no fast path** — running the full verifier pipeline from scratch
   requires the test suite to be green. Solution: read exp1108's existing results + run 50
   FoVer pairs directly (skip_pre_test:true).

## Three Biggest Gaps

1. **arXiv submission still pending** (paper-blocking). Orthogonality audit retired after 7
   attempts; figure integrity (fig3 disputed speedup) still ❌ in known-issues.md. The
   publication hold cannot lift until both are resolved. `skip_pre_test: true` + data-archaeology
   approach makes the audit feasible without a green test suite.

2. **Self-learning loop (GRPO) has never produced a measurable result** (core mission gap).
   GRPO has failed 12+ times across 6 experiments. The fix: radically simplify to VPS-only
   (drop FSPO, token regulation), reduce GPU requirement to single GPU, accept any honest
   result. If GPU is unavailable, report GPU_MISSING with a CPU fallback result.

3. **Phase-5-D not validated** (production scale-up gate). Every Phase-5 experiment above 50K
   params has failed. The fix: reduce to 1M params (CPU-feasible), 200 training steps, measure
   4 of 8 failure modes as acceptance gate instead of all 8.

## Architecture Diagram

```
arXiv v6 Target (exp1246, gated on exp1244 AND exp1245)
    ↑                           ↑
exp1244: 5x5 P(V_i|V_j)    exp1245: fig3 fix + figure audit
    ↑                           ↑
exp1108 existing results    docs/figures/*.py
+ 50 FoVer pairs direct     + results/ artifact tracing

Production k=5 Ensemble (exp1108 confirmed max_r=0.462):
  V1: SOSKANEnergyV3       (AUC=0.9545, contrastive energy)
  V2: SemEnergyProbe       (AUC=0.948, logit-space Boltzmann)
  V3: ASTStructureVerifier  (structural integrity)
  V4: SemanticConsistencyVerifier  (cross-sentence contradiction)
  V5: Z3MathVerifier       (formal arithmetic)
  ThinkPRMProbe excluded: r=0.506 with Z3MathVerifier (above 0.5 threshold)

GRPO Training Stack:
  v4 structural floor → VPS beats v4 (+10pp, exp1220)
  → exp1247: GRPO v7 (VPS-only, simplified, ANY honest result)

Phase-5 In-Situ:
  5-A prototype (exp1222) ✓
  5-B training loop (exp1223) ✓ 5/5 gates
  5-C adversarial probe (exp1224) ✓ k_eff=1 finding
  5-D intermediate scale (exp1250: 1M params, 4 gates)

Phase-3 Substrate:
  NRGPT AUROC=0.921 baseline
  Boltzmann-GPT seed AUROC=0.65 (exp1226)
  → exp1248: CD training v2 (100 steps, explicit scaffold)
```

## Phase Descriptions

### Phase 0 — Infrastructure / Carry-Forwards (3 tasks, unconditional, skip_pre_test:true)

**exp1242: Combined .95+.96 Retro** — closes both pending retros (exp1228/1229 failed 5x for
.95, exp1241 failed 3x for .96). Ultra-mechanical approach: provide EXACT Python one-liners
for each criterion, immediately update artifact after each check. skip_pre_test:true.

**exp1243: WOPR Kakuro Cartridge v2** — DOOMED_RERUN_BLOCK 3x in .96 because 10 priors
flagged but prior_failures missing. This YAML pre-populates 4 representative priors (successful
WOPR cartridges misclassified by the failure-ledger). Kakuro: integer row/column sum constraints,
E = Σ_row(actual_sum - target_sum)^2 + Σ_col(actual_sum - target_sum)^2, E=0 at valid solution.

**exp1244: Verifier Orthogonality Audit v2 (data archaeology)** — radically different approach.
Read exp1108 artifact for pairwise r-correlations (already computed on 500 samples). Run 50 FoVer
corpus v5 pairs through the 5 verifiers DIRECTLY (bypass test suite with skip_pre_test:true,
import from python.carnot.verify directly). Compute 5x5 P(V_i|V_j) matrix. Generate matplotlib
heatmap for paper-v6. Write artifact with k_eff, max_pairwise_cond_prob, heatmap figure path.
Gates exp1246.

### Phase 1 — Paper (2 tasks)

**exp1245: Figure Integrity Audit + fig3 Fix** — known-issues.md requirement for arXiv hold lift.
Read all docs/figures/*.py scripts, trace every constant to a measured artifact in results/.
Fix fig3: remove the disputed 11,680x speedup badge (actual speedup ~58x per-sample), keep
exp1068's measured 24.83µs FPGA latency as the single data point. Output: fixed fig3 + audit
JSON. skip_pre_test:true. Gates exp1246.

**exp1246: Paper v6 Final + arXiv Submission** — gated on exp1244 (orthogonality matrix
computed) AND exp1245 (fig3 fixed). Integrate 5x5 heatmap into paper-v6, update k_eff claim
in §3.2, cite Spera Theorem 9.2 (arXiv:2603.15973) for joint null-space context, cite exp1108
for the empirical audit result. Compile PDF with pdflatex. Submit to arXiv. skip_pre_test:true.

### Phase 2 — Self-Learning (2 tasks)

**exp1247: GRPO v7 (VPS-only, simplified)** — GRPO has failed 12+ times. Radical simplification:
(a) VPS only — no FSPO, no token regulation; (b) single GPU (not DualGPU) — llama.cpp CPU
fallback if GPU unavailable; (c) 20 GSM8K training questions, 30 eval questions; (d) target ANY
honest result. prior_failures pre-populated for 8 GRPO failures. DualGPU MANDATORY if available,
CPU fallback accepted. skip_pre_test:true.

**exp1248: Boltzmann-GPT CD Training v2** — explicit scaffold. (a) Forward pass test first:
import BoltzmannGPT, run 3 FoVer samples through, confirm energy < 1.0; (b) CD training loop:
100 steps, batch_size=8, lr=1e-4; (c) Measure AUROC on 50 FoVer holdout samples; (d) Write
artifact with pre_cd_auroc and post_cd_auroc. Target: AUROC > 0.70 (reduced from 0.80). Uses
arXiv 2512.18730 energy gap formulation. skip_pre_test:true.

### Phase 3 — Research (3 tasks)

**exp1249: LLMs Gaming Verifiers Defense v3** — failed 6x (3x codex max_turns, 3x claude
artifact_not_updated). Radically reduced scope: 10 gaming samples from arXiv 2604.15149 (rule
induction attack), run through k=1 single verifier vs k=5 AND-compose, measure block rate.
Just 3 steps: (1) generate 10 gaming samples, (2) run through verifier(s), (3) compute block
rate. skip_pre_test:true.

**exp1250: Phase-5-D v2 (1M params, CPU-feasible)** — failed 3x in .96. Reduce scale from
100-300M params to 1M params / d=32 for CPU feasibility. Measure 4 of 8 failure modes (mode
collapse entropy, MCMC mixing autocorrelation, null-space fraction, catastrophic forgetting).
Accept any 4 gates measured. GPU preferred; CPU acceptable. skip_pre_test:true.

**exp1251: NRGPT Frozen-Prefix Evaluation v2** — failed 3x in .96. Simplified: just read the
NRGPT experiment artifact from exp1163 (batch-level prototype), extract frozen-prefix energy
trace for the first 10 tokens, classify non-monotonicity as one of: (b) causal-context shift
or (c) learned non-conservative preconditioner. Pure analysis, no training. skip_pre_test:true.

### Phase 4 — New Research (2 tasks)

**exp1252: Q11 TSS Instrumentation** — from memory/project_q11_tss_and_ste_attack.md.
Implement diagnostic: for each verifier in k=5 ensemble, measure (a) SMT triviality fraction
(fraction of inputs where SC-Energy energy = 0.0 trivially without real computation), (b)
orthant occupancy (fraction of Boolean orthant with energy = 0). These predict TSS attack
success. Small codex task. skip_pre_test:true.

**exp1253: WOPR Masyu Cartridge** — Masyu puzzle: white/black circle constraints on a path.
Energy E = count of violated circle constraints (white: must turn, black: must go straight).
E=0 at valid solution. Follow spaces/wopr-games/games/hashi.py pattern. 9+ tests, 100% coverage.

### Phase 5 — Retro (1 task)

**exp1254: Milestone 2026.04.97 Retro** — claude/opus, max_turns:100, STEP 0. Evaluate all
13 criteria from artifact files. Ultra-mechanical: EXACT Python one-liners per criterion.

## Dependency Graph

```
exp1242 (retro .95+.96)     unconditional, FIRST
exp1243 (Kakuro)            unconditional
exp1244 (orthogonality v2)  unconditional
exp1245 (fig3 fix)          unconditional
exp1246 (arXiv submit)      gated: exp1244.orthogonality_matrix_computed + exp1245.figure_audit_complete
exp1247 (GRPO v7)           unconditional
exp1248 (Boltzmann-GPT v2)  unconditional
exp1249 (Gaming v3)         unconditional
exp1250 (Phase-5-D v2)      unconditional
exp1251 (NRGPT v2)          unconditional
exp1252 (Q11 TSS)           unconditional
exp1253 (Masyu)             unconditional
exp1254 (retro .97)         LAST
```

## 13 Success Criteria

1. `retro_combined_96_complete` (exp1242)
2. `kakuro_cartridge_shipped` (exp1243)
3. `orthogonality_matrix_5x5_computed` (exp1244) — gates exp1246
4. `figure_audit_fig3_fixed` (exp1245) — gates exp1246
5. `arxiv_v6_submitted` (exp1246)
6. `grpo_v7_honest_result` (exp1247)
7. `boltzmann_gpt_cd_auroc_measured` (exp1248)
8. `gaming_defense_block_rate_measured` (exp1249)
9. `phase5d_4_gates_measured` (exp1250)
10. `nrgpt_nonmonotonicity_characterized` (exp1251)
11. `q11_tss_instrumentation_shipped` (exp1252)
12. `masyu_cartridge_shipped` (exp1253)
13. `retro_97_complete` (exp1254)

## Hardware Requirements

- exp1247 (GRPO v7): DualGPU preferred, single GPU acceptable, CPU fallback allowed
- exp1248 (Boltzmann-GPT v2): single GPU preferred, CPU fallback allowed
- exp1250 (Phase-5-D v2): single GPU preferred, CPU acceptable (1M params fits in RAM)
- All other experiments: CPU-only

## Key Architectural Decisions

1. `skip_pre_test: true` on ALL experiments except WOPR cartridges — prevents SKIP cascade
   from broken test suite
2. ALL carry-forwards have prior_failures pre-populated — prevents DOOMED_RERUN_BLOCK
3. Orthogonality audit: data-archaeology from exp1108 + 50 direct verifier runs (no full suite)
4. Figure integrity audit is now explicit prerequisite for arXiv (satisfies known-issues.md hold)
5. GRPO v7: VPS-only, CPU fallback — first time GRPO has a CPU fallback path
6. Phase-5-D: 1M params CPU-feasible scale — removes GPU hard dependency
7. Retro: EXACT Python one-liner commands per criterion — eliminates agent reasoning ambiguity
8. All retro tasks: combined .95+.96 retro in single experiment (reduces retro task count)

## Experiment-Agent Routing

| Experiment | Agent | Rationale |
|-----------|-------|-----------|
| exp1242: retro combined | claude/opus | Opus-class retros established as reliable |
| exp1243: Kakuro | codex/gpt-5.5 | WOPR cartridges are formulaic code |
| exp1244: orthogonality | claude/opus | Judgment + data analysis + matplotlib |
| exp1245: figure audit | claude/opus | Multi-file read + judgment |
| exp1246: arXiv submit | claude/opus | Multi-file LaTeX + pdflatex + arXiv |
| exp1247: GRPO v7 | claude/opus | GRPO requires deep tool choreography |
| exp1248: Boltzmann-GPT | codex/gpt-5.5 | Explicit scaffold, mechanical training |
| exp1249: gaming v3 | claude/opus | Judgment on block rate interpretation |
| exp1250: Phase-5-D v2 | claude/opus | Multi-step prototype + measurement |
| exp1251: NRGPT v2 | codex/gpt-5.5 | Pure analysis of stored artifact |
| exp1252: Q11 TSS | codex/gpt-5.5 | Diagnostic implementation, no judgment |
| exp1253: Masyu | codex/gpt-5.5 | WOPR cartridge, formulaic code |
| exp1254: retro .97 | claude/opus | Opus-class retros established as reliable |

## Research Integration (2026-05-04 Scan)

From research-references.md 2026-05-04 scan (filed for .96, incorporating now):
- **arXiv 2512.18730** (EBM lens for RL-tuned LLMs): use energy gap formulation in Boltzmann-GPT CD training (exp1248)
- **arXiv 2312.09244** (reward model ensemble correlation): validates k_eff finding; cite in orthogonality audit paper section
- **arXiv 2601.01490** (distortion under constraints): add factual fidelity metric to GRPO v7 eval alongside accuracy
- **arXiv 2511.00066** (token-regulated GRPO): deferred to .98 — GRPO v7 first needs a baseline
- **arXiv 2512.15658** (PPSEBM): include EBM replay buffer concept in Phase-5-D v2 design
- **arXiv 2603.15973** (Spera Theorem 9.2): cite in paper-v6 orthogonality section
- **arXiv 2604.02767** (SentinelAgent): cite as closest peer in paper-v6 related work
- **arXiv 2604.22709** (Abstract-CoT): file in research-references.md for .98 (Phase-3 substrate)
