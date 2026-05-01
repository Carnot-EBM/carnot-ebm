# Research Roadmap v86: Failure-Ledger v2 + Phase 1a Unblocked + Verifier Diversity + FPGA Sequential Sampler + arXiv Submission

**Milestone:** 2026.04.86
**Planned:** 2026-05-01
**Target Wall Time:** ~600 min
**Experiments:** exp1104 – exp1114 (11 experiments)

---

## What Milestone .85 Proved

Milestone .85 completed 13/14 criteria (92.9%) — the strongest recovery after .84's 4/13.
The single unmet criterion was **Phase 1a adversarial verifier robustness audit (exp1092)**, blocked
for the **third consecutive milestone** by the conductor's coarse keyword matcher generating 18
false-positive prior-failure gate matches.

Key substantive findings from .85:

1. **GSM8K extraction root cause fixed** (exp1101): VeriCoT TP rate 0→1.0 after adding
   `_EQ_INLINE_RE` pattern to vericot_validator.py. SOTA models write equation-style CoT
   (A OP B = C) rather than prose; old regex missed them.

2. **Position paper v2 arXiv-ready** (exp1091): 7113 words, 5 figure scripts, arxiv-metadata.yaml.
   Submission target: 2026-05-15 (14 days from now — **CRITICAL DEADLINE**).

3. **SemEnergy probe AUROC=0.948 at 0.017ms/example** (exp1096): 100× faster than SOS-KAN v3
   while matching its AUROC. Tier 0c upgrade confirmed viable.

4. **Phase 1c null-space measurement** (exp1093): `joint_null_space_fraction=0.0` (below
   threshold) but `and_composition_viable=False` — max pairwise r-correlation=0.656 > 0.5
   threshold. AND-composition does not shrink the kernel exponentially because the current
   verifier suite (SOS-KAN, IsingEnergy, SemEnergy) is too correlated. Requires 6+ diverse
   verifiers.

5. **Phase 2a FPGA sampler mismatch confirmed empirically** (exp1094): KL(FPGA||Gibbs)=3.07
   (threshold 0.05 — far above). Root cause: synchronous parallel Glauber loses detailed
   balance on frustrated antiferromagnetic ring. KV260 bitstream must be redesigned with
   sequential single-site updates.

6. **RLVR+SSD honest negative** (exp1099): energy_all_zero=True — the corpus had been
   pre-filtered by AND-compose-k5, making all energy scores 0.0. Carnot differentiation
   not observable on this degenerate corpus.

7. **Cascade depth SOTA finding** (exp1100): SOTA outputs need mean_cascade_depth=2.20 vs
   FoVer corpus 1.08. SOTA outputs are harder to verify — expected behavior, but informs
   threshold tuning.

---

## Three Biggest Gaps to .86

### Gap 1: Phase 1a adversarial verifier robustness audit — THIRD CONSECUTIVE BLOCK

The single unmet .85 criterion. Root cause in both .84 and .85: the conductor's keyword matcher
matched "adversarial" + "verifier" to 18 prior experiments and generated gate-check failures
because prior_failures was not declared for all 18. The fix has two parts:
- **Structural fix** (exp1104): tighten the keyword matcher to require ≥2 scope-vocabulary keywords
  so "verifier robustness audit" doesn't inherit counts from unrelated "step verifier" experiments.
- **Planner fix** (exp1106): declare prior_failures for ALL 18 experiments from exp1092's
  gates_evaluated list. All 18 IDs are now known from the blocked artifact JSON.

### Gap 2: Verifier ensemble too correlated for AND-composition

Phase 1c (exp1093) measured max pairwise r-correlation=0.656 > 0.5. AND-composition assumes
exponential kernel shrinkage per additional orthogonal verifier. With highly correlated verifiers,
it does not shrink. Need 6+ verifiers with genuinely orthogonal error modes. The current ensemble
(SOS-KAN, IsingEnergy, SemEnergy) shares a common failure pattern: all three rely on token-level
statistics. Need structural diversity:
- Z3-based symbolic math verifier (checks arithmetic correctness via formal solver)
- AST-based code structure verifier (checks syntactic/structural properties)
- These have near-zero r-correlation with current ensemble by construction

### Gap 3: RLVR+SSD corpus degenerate (energy_all_zero)

Exp1099 failed because AND-compose-k5 pre-filtered the training corpus to all-zero energy examples.
The fix is to use a corpus where some outputs have genuine energy > 0 — i.e., run SOTA model live
on fresh questions WITHOUT pre-filtering by AND-compose. Apply top-k energy selection (select the
30th percentile by energy as "clean" training signal) rather than median threshold.

---

## Milestone Architecture

```
Phase 0: Failure-Ledger v2 Infrastructure (MANDATORY — conductor surgery)
         exp1104 ──┐  exp1105 ──┐
                   │             │
Phase 1:           ↓             ↓
         exp1106 Phase 1a Adversarial Audit v2
         [gated on exp1104.failure_ledger_id_fix_deployed]

Phase 2: Verifier Diversity Expansion
         exp1107 New Diverse Verifiers (codex)
              ↓
         exp1108 Ensemble Diversity v2 [gated on exp1107]

Phase 3: FPGA Redesign
         exp1109 KV260 Sequential Glauber v3 (opus, hardware)
         [prior_failures: exp1094 KL=3.07 parallel Glauber]

Phase 4: Live GPU + RLVR+SSD
         exp1110 RLVR+SSD v2 Non-Degenerate (GPU, SOTA model)
         [prior_failures: exp1099 energy_all_zero]

Phase 5: Self-Learning + Goodfire
         exp1111 ThinkPRM v2 Retrain on 7349 PRM examples (GPU)
         exp1112 LLM Failure Exemplar Corpus v1 + Goodfire TP (codex)

Phase 6: arXiv Submission
         exp1113 arXiv LaTeX Bundle + GitHub Pages

Phase 7: Retro
         exp1114 Milestone 2026.04.86 Retrospective
```

---

## Dependency Graph

```
exp1104 (FL-v2 Issues 1+5+manifest) ─────→ exp1106 (Phase 1a)
exp1105 (FL-v2 Issues 2+3+4)

exp1107 (New Diverse Verifiers) ──────────→ exp1108 (Ensemble Diversity v2)
                                              ↓
                                          exp1110 (RLVR+SSD v2) [soft dependency]

exp1109 (KV260 v3) — standalone

exp1111 (ThinkPRM v2) — standalone (GPU)

exp1112 (Failure Exemplar) — standalone

exp1113 (arXiv Bundle) — standalone (priority: CRITICAL, deadline 2026-05-15)

exp1114 (Retro) — final
```

---

## Phase Descriptions

### Phase 0: Failure-Ledger v2 Infrastructure (MANDATORY)

**Context:** Known-issues.md MANDATORY-NEXT-MILESTONE entry. Without these conductor
fixes, .86 will hit identical walls to .84 and .85.

**exp1104 — Failure-Ledger v2 Issues 1+5+manifest**
Fixes three conductor regressions in a single surgery pass:
- Issue 1: Count failures by `experiment_id` (not title-prefix) — prevents false inheritance
- Issue 5: Tighten `is_doomed_rerun()` to require ≥2 scope-vocabulary keyword overlap —
  prevents bulk false-positive matching
- Manifest enforcement: check `ops/exclusion_manifest.yaml` at dispatch time, not only at
  planning time — stops exp906 from appearing as slowest experiment (3rd consecutive milestone)

**exp1105 — Failure-Ledger v2 Issues 2+3+4**
Fixes three additional conductor regressions:
- Issue 2: Reset 3-fail cap when a fix-commit lands between attempts (prevents cap race)
- Issue 3: Stable-deliverable detection requires `mtime > task_start_time` (prevents false
  stale-artifact kills like exp1090's Opus agent being killed before writing)
- Issue 4: Pre-test fingerprint cache saves END state (prevents cache-miss regressions
  when operator commits mid-pre-test)

### Phase 1: Phase 1a Adversarial Verifier Robustness Audit (3rd attempt — MUST SUCCEED)

**exp1106 — Phase 1a adversarial verifier robustness audit v2**
- All 18 prior_failures declared explicitly (from exp1092 blocked artifact)
- Gated on exp1104.failure_ledger_id_fix_deployed so the id fix is live before gate-check fires
- Attack corpus: 100 adversarial examples using APRM-style attack patterns (arXiv 2511.22888):
  stylistic padding, structured formatting, IPT isomorphic perturbation, verbosity inflation,
  confidence signaling
- Measures false-pass rate per attack type (target: max < 5%)
- Uses NullSpaceEstimator from diagnostics library (exp1090)
- Acceptance: phase1a_false_pass_below_5pct=True

### Phase 2: Verifier Diversity Expansion

**exp1107 — New diverse verifiers v1 (codex)**
Adds 3 new verifiers with structurally orthogonal kernels to current ensemble:
1. `Z3MathVerifier`: Uses Z3 SMT solver to check arithmetic constraints in model CoT.
   Kernel: numeric correctness via formal solver → orthogonal to SOS-KAN token statistics.
2. `ASTStructureVerifier`: Checks code AST structural properties (valid syntax, no obvious
   runtime errors). Kernel: structural/syntactic → orthogonal to both SOS-KAN and Z3.
3. `SemanticConsistencyVerifier`: Checks logical consistency across CoT steps via simple
   contradiction detection. Kernel: cross-sentence logic → orthogonal to token statistics.

Each verifier must follow the `SamplerBackend` protocol in python/carnot/verify/.

**exp1108 — Ensemble diversity re-measurement + AND-composition viability**
Re-runs NullSpaceEstimator with 6-verifier ensemble (SOS-KAN + ThinkPRM + SemEnergy + Z3Math +
AST + SemanticConsistency). Target: max pairwise r-correlation < 0.5 (and_composition_viable=True).
If achieved, AND-composition with k=5 is validated for DBAE-EBM Stage 3.

### Phase 3: FPGA KV260 Redesign

**exp1109 — KV260 Ising sampler v3 — sequential single-site updates**
Root cause (exp1094): synchronous parallel Glauber violates detailed balance on frustrated
antiferromagnetic ring (KL=3.07). Fix: redesign Verilog to use sequential single-site updates
(sweep one spin at a time in round-robin order) as recommended by arXiv 2603.25910 and
arXiv 2604.01564. Alternatively: implement time-multiplexed p-bit reuse (arXiv 2604.01564)
which achieves stable synchronous operation at lower hardware cost.

Target: KL(FPGA||Gibbs) < 0.05 (vs current 3.07). If KV260 board is unreachable, produce
Python simulation proving the redesigned update scheme achieves KL < 0.05 and write the
synthesizable Verilog.

Stretch goal: if KV260 board is reachable and bitstream synthesized, also attempt Potts q=3
bitstream deployment (builds on exp1098's validated Verilog).

### Phase 4: RLVR+SSD with Non-Degenerate Corpus (Live GPU)

**exp1110 — RLVR+SSD v2 with SOTA live GPU inference**
Root cause (exp1099): corpus pre-filtered by AND-compose-k5 to all-zero energy. Fix:
1. Run Qwen3.6-35B-A3B-GGUF on 200 fresh GSM8K questions (live GPU, DualGPURunner)
2. Compute raw per-verifier energy scores WITHOUT AND-compose pre-filtering
3. Select training corpus: top-30th percentile by energy (high-energy = more violations = harder examples)
4. Apply RLVR training: learn to avoid high-energy outputs
5. Apply SSD: low-energy outputs as teacher signal for student distillation

Use DualGPURunner (both RTX 3090s) for SOTA model inference to break the 18th consecutive
DualGPU idle milestone.

Target: improvement_over_baseline > 0.0 with honest_verdict="positive_result" or
"honest_negative_with_non_degenerate_corpus".

### Phase 5: Continuous Self-Learning + Goodfire Exemplar Corpus

**exp1111 — ThinkPRM v2 retrain on 7349-example PRM corpus (Tier 3 self-learning)**
The ThinkPRM (AUROC=0.9885) was trained on a smaller corpus. Exp1084 generated 7349 step-level
PRM training examples (3.7× the original corpus). Retrain ThinkPRM on this expanded corpus.
Target: AUROC ≥ 0.995 (beating current 0.9885). This is the Tier 3 continuous self-learning
experiment required by research-program.md: the verifier improves as more labeled data accumulates.

Satisfies: research-program.md "Include at least one self-learning experiment every milestone."

**exp1112 — LLM failure exemplar corpus v1 + Goodfire Silico comparison (codex)**
Build `data/llm_failure_exemplars.jsonl` with ≥30 named, reproducible failure modes:
- Goodfire Silico published exemplars: 9.11>9.9, trolley problem, deceptive disclosure
- Carnot project-internal findings from FoVer corpus
Feed each through Carnot's cascade and report TP rate per verifier tier.
Validates: mathematical-objective tier (Z3) should catch 9.11>9.9 trivially.
Answers: does Carnot's engineering tier beat the "alchemy precision" critique?

### Phase 6: arXiv Submission (CRITICAL DEADLINE 2026-05-15)

**exp1113 — arXiv LaTeX bundle + GitHub Pages launch**
Position paper v2 (docs/position-paper-draft-v2.md, 7113 words) is arXiv-ready. This experiment:
1. Converts Markdown → LaTeX (pandoc + bibliography)
2. Compiles 5 matplotlib figures to PDF
3. Assembles arXiv submission bundle (main.tex, figures/, carnot.bib)
4. Validates arXiv metadata (docs/arxiv-metadata.yaml)
5. Pushes docs/index.html (landing page) with updated position paper link
Target: submission_bundle_complete=True with PDF validated by pdflatex.

### Phase 7: Retro

**exp1114 — Milestone 2026.04.86 retrospective**
Standard 14-criterion evaluation. max_turns: 20.

---

## Success Criteria (12)

| # | Criterion | Experiment |
|---|-----------|------------|
| 1 | failure_ledger_id_fix_deployed | exp1104 |
| 2 | failure_ledger_mtime_cap_manifest_deployed | exp1105 |
| 3 | phase1a_false_pass_below_5pct | exp1106 |
| 4 | new_diverse_verifiers_deployed_3_verifiers | exp1107 |
| 5 | and_composition_viable_r_corr_below_05 | exp1108 |
| 6 | kv260_v3_kl_measured_below_threshold | exp1109 |
| 7 | rlvr_ssd_v2_non_degenerate_honest_result | exp1110 |
| 8 | thinkprm_v2_auroc_above_099 | exp1111 |
| 9 | llm_failure_exemplar_corpus_30_exemplars | exp1112 |
| 10 | goodfire_cascade_tp_rate_measured | exp1112 |
| 11 | arxiv_bundle_complete | exp1113 |
| 12 | retro_complete | exp1114 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|------------|----------|-------|
| exp1104 | CPU only | Conductor surgery — no GPU |
| exp1105 | CPU only | Conductor surgery — no GPU |
| exp1106 | CPU only | Energy scoring with existing verifiers |
| exp1107 | CPU only | Verifier implementation |
| exp1108 | CPU only | NullSpaceEstimator evaluation |
| exp1109 | KV260 (FPGA) | If board unreachable: CPU Python sim |
| exp1110 | 2× RTX 3090 (DualGPU) | SOTA model live inference — DualGPURunner MANDATORY |
| exp1111 | 1× RTX 3090 | ThinkPRM training |
| exp1112 | CPU only | Data collection + cascade scoring |
| exp1113 | CPU only | LaTeX compilation |
| exp1114 | CPU only | Retro analysis |

---

## Key Architectural Decisions for .86

1. **No gemini agent_type** (rate-limited per known-issues.md): all long-context via Opus.
2. **Codex re-enabled** (fixed in .85): exp1107, exp1112 routed to codex.
3. **DualGPU MANDATORY for exp1110**: 18th consecutive idle milestone; exp1110 requires
   DualGPURunner as a hard constraint in the prompt.
4. **Failure-ledger id fix must run before Phase 1a**: exp1106 gated on exp1104.
5. **ALL 18 prior_failures declared in exp1106**: complete list from exp1092 blocked artifact.
6. **arXiv deadline 2026-05-15 is HARD**: exp1113 is unconditional and high-priority.
7. **Model coherence rule (from known-issues.md P2)**: no `model: opus` on `agent_type: codex`
   tasks. Codex tasks use default codex model (gpt-5.5).

---

## New Papers Incorporated (from .86 planning arxiv scan)

| arXiv ID | Title | Used In |
|----------|-------|---------|
| 2604.01564 | Unified Performance–Cost Landscape of Parallel p-bit Ising Machines | exp1109 (FPGA redesign) |
| 2603.25910 | Finite-Time Observability of Oscillatory Instabilities in Synchronous p-bit Dynamics | exp1109 + position paper |
| 2511.22888 | Adversarial Training for Process Reward Models (APRM) | exp1106 (attack patterns) |
| 2604.13602 | Reward Hacking in the Era of Large Models | position paper Section 3 |
| 2601.20802 | Reinforcement Learning via Self-Distillation (SDPO) | exp1110 (training algorithm) |
| 2504.01005 | When To Solve, When To Verify: Compute-Optimal GenRM | position paper Section 5 |

---

## Process Improvements from .85 Retro Applied to .86 Design

| .85 Bottleneck | .86 Design Response |
|----------------|---------------------|
| Phase 1a blocked 3rd consecutive (planner) | 18 prior_failures ALL declared; failure-ledger id fix gates exp1106 |
| exp906 manifest regression (3rd) | exp1104 includes dispatch-time manifest enforcement fix |
| stable-deliverable false-positive killed exp1090 | exp1105 includes mtime>task_start_time fix |
| DualGPU 18th idle | exp1110 has DualGPURunner as hard constraint |
| sequential fast experiments (6 experiments serialized) | exp1104+1105 run in parallel; exp1107+1109 run in parallel |
| doc reconciliation 28min blocking | exp1105 includes doc batching improvement |

---

## Estimated Wall Time

| Phase | Experiments | Est. Time |
|-------|------------|-----------|
| Phase 0: FL-v2 infra | exp1104+1105 (parallel) | 45 min |
| Phase 1: Phase 1a audit | exp1106 | 45 min |
| Phase 2: Verifier diversity | exp1107+1108 | 55 min |
| Phase 3: FPGA redesign | exp1109 | 90 min |
| Phase 4: RLVR+SSD | exp1110 | 80 min |
| Phase 5: Self-learning | exp1111+1112 (parallel) | 75 min |
| Phase 6: arXiv bundle | exp1113 | 60 min |
| Phase 7: Retro | exp1114 | 20 min |
| **Total** | **11 experiments** | **~470 min** |
