# Research Roadmap — Milestone 2026.04.85

**Milestone:** 2026.04.85
**Title:** Phase Discipline Infrastructure + Position Paper arXiv + Gate-Blocked Recovery
**Experiments:** 1090–1103 (14 experiments)
**Planned Date:** 2026-05-01
**Target Wall Time:** ~490 min
**Target Criteria:** 14/14

---

## What Milestone .84 Proved

**4 of 13 criteria met (31%).** Three genuine achievements:

1. **FR-11 alpha_t=0.38 confirmed live on SOTA MoE (exp1077):** Qwen3.6-35B-A3B-GGUF on
   2× RTX 3090 yielded alpha_t=0.38, lower than the 0.8B baseline (0.78) because SOTA models
   are harder for the AND-composed verifier to distinguish from temperature filtering. Result is
   honest; model_tier='sota_moe'; fr11_loop_closed=true.

2. **First-ever positive live benchmark result with SOTA IT model (exp1079):**
   HumanEval pass@1 improved 0→36% after Carnot correction on Qwen3.6-35B-A3B.
   GSM8K extraction failed (TP=0) for the second consecutive milestone — code verification
   works; math reasoning extraction is the remaining bottleneck.

3. **Step-level PRM dataset at scale (exp1084):** 7349 step-labeled examples generated
   (3.7× above the 2000 target). ThinkPRM retrained on 300-sample subset (AUROC 0.9885→0.7929)
   — full retrain on 7349 examples is deferred to .85.

**Primary bottleneck (operational):** 6 of 13 planned experiments (46%) were blocked by
conductor gate enforcement because the .84 planner omitted required `prior_failures`
declarations in the roadmap YAML. The conductor's gate-check is working as designed — the
gap is at the planner layer. The .85 planner has scanned research-complete.yaml exhaustively
and declared all required prior_failures entries.

**Gemini deprecation:** exp1078 (position paper) and exp1087 (gemini conductor) both
relied on gemini-3.1-pro-preview, which was suspended due to 429 rate limits. The position
paper arXiv deadline (2026-05-15) is now critically close. .85 routes position paper
through Opus.

---

## Three Biggest Gaps

1. **Position paper arXiv deadline (2026-05-15):** The paper has not been written (exp1078
   never ran in .84). With two weeks until the arxiv target, this is the most time-critical
   deliverable. Route through Opus (no gemini), max_turns:80.

2. **Phase Discipline Infrastructure (5 MANDATORY tasks, all 3+ milestones overdue):**
   CLAUDE.md mandates every phase must have: (a) prototype, (b) empirical pass/fail criteria,
   (c) adversarial check before scaling. Known-issues.md has listed 5 specific tasks since
   2026-04-30. The .85 planner MUST deliver all 5:
   - Diagnostic instrumentation library (cross-cutting)
   - Phase 1a: Adversarial verifier robustness audit
   - Phase 1c: Verifier joint null-space measurement
   - Phase 2a: Sampler correctness audit (KV260 + GPU baseline)
   - Phase 3a: Pre-prototype adversarial round (DBAE-EBM threat model)

3. **Gate-blocked carry-forwards (6 experiments, prior_failures now declared):** SemEnergy
   probe, N-Queens cartridge, Potts machine RTL, RLVR+SSD integration, cascade validation
   on SOTA outputs, HF gallery update — all technically ready, just needed prior_failures YAML.

---

## Architecture Context

```
Phase 1 (verify-repair) — WORKING
  └── Tier 0a: ThinkPRM step verifier (AUROC 0.9885)
  └── Tier 0b: SpilledEnergy (logit-space)
  └── Tier 0c: NUP Probe / SemEnergy (Tier 0c upgrade — .85 target)
  └── Tier 2: SOS-KAN v3 (AUROC 0.9545)
  └── Tier 3: Ising EBM (23.4μs on KV260 FPGA)
  └── Repair: VerifyRepairPipeline → HumanEval +36% (first SOTA positive)

Phase 2 (hardware) — POC VALIDATED
  └── KV260 FPGA: 24.83μs at 64 spins (board unreachable in .84)
  └── Sampler correctness: synchronous parallel Glauber caveat documented
  └── Future: Extropic Z1 (early access 2026), photonic platforms

Phase 3 (foundation model) — ARCHITECTURE DERIVED, PROTOTYPE PENDING
  └── DBAE-EBM: Deterministic Bounded Autoencoder + Latent EBM
  └── Phase 3a adversarial round REQUIRED before writing prototype code
  └── Phase 3→7 defence-layer stack: derivation complete (docs/research-notes/)

FR-11 (self-improvement) — LOOP CLOSED
  └── alpha_t=0.38 on SOTA MoE (confirmed live GPU)
  └── 7349 step-level PRM training examples generated
  └── RLVR+SSD combination pending (.85 carry-forward)
```

---

## Phase Design

### Phase 0 — Diagnostic Infrastructure (MANDATORY, unconditional)

**exp1090** — Diagnostic Instrumentation Library
- Implements the shared Python module for all phase prototype validation:
  α_t tracking, KL divergence, joint null-space estimation, decoded-text diversity,
  manifold-coverage metrics. Used by exp1092/1093/1094.
- Also: prior-failures scan of research-complete.yaml to validate .85 YAML declarations.
- Deliverable: `python/carnot/eval/diagnostics.py` + scan report
- Priority: infrastructure (counts as reserved slot), model: opus, max_turns: 35

### Phase 1 — Position Paper (CRITICAL — deadline 2026-05-15)

**exp1091** — Position Paper v2 arXiv Prep (Opus route)
- Read ALL source material: draft v1, all research-notes/, all key result JSONs.
- Produce arXiv-ready v2 with 5 matplotlib figure scripts, complete references,
  technical review of theorems, arXiv metadata YAML.
- prior_failures: exp1078 (gemini-3.1-pro-preview 429-rate-limited, never ran)
- Standalone (parallel with Phase 0)
- model: opus, max_turns: 80

### Phase 2 — Mandatory Phase Discipline (4 MANDATORY infrastructure tasks)

**exp1092** — Phase 1a: Adversarial Verifier Robustness Audit
- Measure false-pass rate on adversarially-crafted LLM outputs.
  Use attack patterns from arXiv 2603.06621 (stylistic shortcuts, IPT from arXiv 2604.15149).
  Acceptance: false-pass rate < 5% across canonical attack patterns.
- gated_on: exp1090.diagnostics_library_written
- max_turns: 40

**exp1093** — Phase 1c: Verifier Joint Null-Space Measurement
- Empirically measure dim(∩_i ker E_i) for existing 4-6 verifiers.
  Use r-correlation metric from arXiv 2604.12086 alongside algebraic null-space estimator.
  Acceptance: joint null-space dimension < 5% of input space.
- gated_on: exp1090.diagnostics_library_written
- max_turns: 40

**exp1094** — Phase 2a: Sampler Correctness Audit
- KV260 board reconnect diagnostic + KL divergence between FPGA samples
  and correct CPU Gibbs samples on a deliberately frustrated J matrix.
  Confirms or refutes Finding #2 (synchronous parallel Glauber non-equilibrium).
  Adds GPU Ising baseline (onnxruntime-gpu CUDA EP, 2× RTX 3090).
- model: opus (hardware integration), max_turns: 40
- Standalone (theoretical fallback if board unreachable)

**exp1095** — Phase 3a: DBAE-EBM Pre-Prototype Adversarial Round
- BEFORE any DBAE-EBM prototype code, run hostile-reviewer analysis:
  identify ways prototype could silently pass acceptance gates without working.
  Failure modes to document: degenerate identity encoder, decoder ignoring bottleneck,
  EBM converging to constant, joint null-space regression.
- Output: threat model document + instrumentation checklist for DBAE-EBM prototype.
- max_turns: 30

### Phase 3 — Gate-Blocked Carry-Forwards (with complete prior_failures)

**exp1096** — SemEnergy Probe v1
- Implements Boltzmann-inspired logit-space energy probe (arXiv 2508.14496 upgrade).
- prior_failures: exp772 (semantic_energy_below_baseline), exp1080 (blocked_gate_check)
- max_turns: 40

**exp1097** — WOPR N-Queens Cartridge
- 8-Queens Ising ground-state solver as WOPR game cartridge.
- prior_failures: exp1070 (cartridge_shipped GTW), exp1071 (cartridge_shipped Lights Out),
  exp1086 (blocked_gate_check — N-Queens scope)
- agent_type: codex (formulaic Ising CSP cartridge), max_turns: 35

**exp1098** — Potts Machine q=3 Verilog + Python Simulation
- q=3 Potts sampler Python simulation + RTL for KV260.
- prior_failures: exp534 (no_advantage on CPU sim), exp1082 (blocked_gate_check)
- agent_type: codex (RTL boilerplate), model: opus, max_turns: 40

**exp1099** — RLVR+SSD Integration v1
- Three-way comparison: RLVR-only, SSD-only, RLVR+SSD on 200 GSM8K examples.
  Uses data/fr11_zenil_distill_v2.jsonl + arXiv 2601.18734 on-policy SSD variant.
  Standalone run; GPU fine-tuning optional (selection mode if GPU unavailable).
- prior_failures: exp467, exp927, exp955, exp978, exp990, exp1014, exp1044, exp1083
- max_turns: 40

**exp1100** — Cascade Validation on SOTA Model Outputs
- Full cascade tier breakdown (0a→0b→2→3) on 100 real Qwen3.6-35B outputs from exp1079.
- prior_failures: exp432, exp453, exp485, exp657, exp705, exp778, exp784, exp882,
  exp946, exp1085
- gated_on: exp1079.humaneval_net_improvement > 0 (confirmed true from .84)
- max_turns: 35

### Phase 4 — Research Breakthroughs

**exp1101** — GSM8K Extraction Diagnostic + VeriCoT Fix
- Root-cause diagnosis: why does VeriCoT get TP=0 on live GSM8K math reasoning
  for second consecutive milestone? Implement NSVIF/Z3 approach or LLM-as-extractor
  alternative. Test on 20 known-wrong GSM8K answers.
- max_turns: 40

### Phase 5 — Gallery

**exp1102** — HF Spaces Gallery Update
- Deploy N-Queens cartridge (if exp1097 succeeded) to live WOPR gallery.
  Decouple from N-Queens cartridge per .84 retro: also deploy any existing
  cartridges that haven't been published yet.
- gated_on: exp1097.nqueens_cartridge_shipped
- max_turns: 25

### Phase 6 — Retrospective

**exp1103** — Milestone 2026.04.85 Retrospective
- Evaluate all 14 success criteria.
- max_turns: 20

---

## Dependency Graph

```
exp1090 (diagnostic library) ──────┬──> exp1092 (Phase 1a adversarial)
                                   └──> exp1093 (Phase 1c null-space)
exp1091 (position paper)           [standalone]
exp1094 (Phase 2a sampler)         [standalone, fallback if board unreachable]
exp1095 (Phase 3a adversarial)     [standalone]
exp1096 (SemEnergy)                [standalone, prior_failures declared]
exp1097 (N-Queens)                 ──> exp1102 (HF gallery, gated)
exp1098 (Potts RTL)                [standalone]
exp1099 (RLVR+SSD)                 [standalone, prior_failures declared]
exp1100 (cascade validation) ──────< exp1079.humaneval_net_improvement>0 [.84 artifact, CONFIRMED]
exp1101 (GSM8K extraction fix)     [standalone]
exp1103 (retro)                    [last, reads all]
```

---

## Success Criteria

| # | Criterion | Experiment | Key Metric |
|---|-----------|-----------|------------|
| 1 | diagnostics_library_written | exp1090 | diagnostics_library_written == true |
| 2 | position_paper_arxiv_ready | exp1091 | arxiv_metadata_written == true |
| 3 | phase1a_false_pass_below_5pct | exp1092 | adversarial_false_pass_rate < 0.05 |
| 4 | phase1c_null_space_below_5pct | exp1093 | joint_null_space_fraction < 0.05 |
| 5 | phase2a_sampler_validated | exp1094 | honest_verdict != "failed" |
| 6 | phase3a_threat_model_written | exp1095 | threat_model_written == true |
| 7 | semenergy_probe_auroc_above_07 | exp1096 | semenergy_auroc > 0.70 |
| 8 | nqueens_cartridge_shipped | exp1097 | final_energy == 0.0 |
| 9 | potts_sim_validated | exp1098 | python_sim_validated == true |
| 10 | rlvr_ssd_honest_result | exp1099 | honest_verdict != "failed" |
| 11 | cascade_validated_sota_outputs | exp1100 | n_outputs_run >= 50 |
| 12 | gsm8k_extraction_fixed | exp1101 | extraction_tp_rate > 0.0 |
| 13 | gallery_updated_hf_spaces | exp1102 | gallery_updated == true |
| 14 | retro_complete | exp1103 | this artifact |

---

## Hardware Requirements

| Experiment | GPU | FPGA | CPU |
|-----------|-----|------|-----|
| exp1090 | No | No | Yes |
| exp1091 | No | No | Yes (Opus) |
| exp1092 | No | No | Yes |
| exp1093 | No | No | Yes |
| exp1094 | Optional | Yes (KV260) | Fallback |
| exp1095 | No | No | Yes |
| exp1096 | No | No | Yes |
| exp1097 | No | No | Yes (codex) |
| exp1098 | No | No | Yes (codex+opus) |
| exp1099 | Optional | No | Yes (selection mode) |
| exp1100 | No | No | Yes |
| exp1101 | Optional | No | Yes |
| exp1102 | No | No | Yes |
| exp1103 | No | No | Yes |

---

## Key Architectural Decisions for This Milestone

1. **No gemini agent type** — user directive 2026-05-01: gemini-3.1-pro-preview 429-throttles.
   Route all long-context work through Opus.

2. **Codex re-enabled** — issue fixed 2026-05-01. Use agent_type: codex for formulaic code:
   N-Queens cartridge (exp1097) and Potts RTL (exp1098).

3. **Prior_failures exhaustively declared** — all carry-forward experiments include
   complete prior_failures YAML from research-complete.yaml scan.

4. **Position paper is MANDATORY first (parallel with infrastructure)** — arXiv deadline
   2026-05-15 leaves no room for another blocked milestone.

5. **Self-learning experiment** — exp1099 (RLVR+SSD) advances continuous self-learning
   Tier 3 (Zenil α_t grounding + RLVR verifier signal).

---

## New arxiv Papers Incorporated

Added to research-references.md (2026-05-01 scan):

| arXiv ID | Title | Incorporated In |
|----------|-------|-----------------|
| 2603.06621 | Reward Under Attack: PRM Robustness | exp1092 attack suite |
| 2604.15149 | LLMs Gaming Verifiers (IPT probe) | exp1092/1093 null-space |
| 2601.18734 | Self-Distilled Reasoner (on-policy) | exp1099 SSD variant |
| 2510.23972 | Efficient Hardware EBM Architecture (Extropic) | exp1091 position paper |
| 2508.17440 | Photonic KAN+Ising Platform | exp1091 position paper |
| 2603.03305 | Draft-Conditioned Constrained Decoding | exp1101 repair quality |
| 2604.12086 | Robust Optimization for Reward Hacking | exp1093 r-correlation |

---

## Expected Outcome

With proper prior_failures declarations and no gemini dependency, the primary
systemic blocker from .84 is eliminated. 14 experiments are achievable with:
- 5 infrastructure tasks satisfying CLAUDE.md phase-discipline mandate
- Position paper on track for arXiv 2026-05-15
- All 6 gate-blocked carry-forwards unblocked via prior_failures YAML
- First honest diagnostic on GSM8K extraction failure (2 milestones broken)
- HumanEval +36% follow-up analysis in exp1100 cascade tier breakdown
