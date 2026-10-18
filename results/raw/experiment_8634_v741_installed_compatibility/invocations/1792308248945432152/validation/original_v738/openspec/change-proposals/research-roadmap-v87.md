# Research Roadmap v87: arXiv Submission + Energy Inversion Fix + GRPO Energy PRM + k=5 Production + FPGA v4

**Milestone:** 2026.04.87
**Planned:** 2026-05-01
**Target Wall Time:** ~500 min
**Experiments:** exp1116 – exp1126 (11 experiments)

---

## What Milestone .86 Proved

Milestone .86 completed 11/12 criteria (91.7%) — second consecutive above 90%.

Key substantive findings:

1. **Phase 1a adversarial audit unblocked** (exp1106): 0% false-pass rate across 5 APRM-style
   attack types (stylistic padding, formatting, IPT isomorphic perturbation, logically corrupted
   reasoning, gradient-style shortcuts). After 3 consecutive blocked attempts, Phase-1a is
   finally certified. The mathematical-objective verifier tier (Z3MathVerifier) was the most
   robust; the learned tier (SOS-KAN) was more susceptible to stylistic attacks — this
   distinction is now documented in the position paper v3.

2. **Failure-Ledger v2 all 4 issues fixed** (exp1104/1105): id-aware failure counting, cap reset
   on fix-commit, mtime-based stale artifact detection, end-fingerprint cache. 14 tests pass.
   The dispatch-time manifest enforcement was explicitly added, BUT exp906 still appeared in
   the .86 milestone (4th consecutive) because the fix deployed DURING the milestone after
   exp906 had already been queued. .87 is the first milestone where this fix will be active
   from the start.

3. **AND-composition k=5 empirically validated** (exp1108): [SOSKANEnergyV3, SemEnergyProbe,
   ASTStructureVerifier, SemanticConsistencyVerifier, Z3MathVerifier] ensemble achieves
   max_r=0.462 < 0.5 threshold. ThinkPRMProbe×Z3MathVerifier pair at r=0.507 is the rogue pair;
   k=6 not viable. Phase-1d target updated to k=5. This ensemble is not yet wired as the
   production default in VerifyRepairPipeline.

4. **KV260 v3 sequential sampler validated in simulation** (exp1109): KL(v3||Gibbs)=0.025 nats
   (threshold 0.05), well below acceptance gate. Hardware deployment deferred — Vivado not on
   PATH. Board reachable at 192.168.51.98. The v4 sparse K=16 + E-MVL + inertia Verilog
   (ising_sampler_v4.v + spec) is written but not yet Python-simulated or synthesized.

5. **RLVR+SSD v2 honest negative** (exp1110): improvement_over_baseline=-0.0004 on non-degenerate
   corpus (energy_distribution_nonzero_fraction=1.0, mean=0.505, stdev=0.257). The 3rd
   consecutive honest negative. Root diagnosis: RLVR+SSD architecture is the wrong training
   loop — it uses energy as a binary signal rather than leveraging ThinkPRM v2's calibrated
   AUROC=0.9946 reward quality.

6. **ThinkPRM v2 retrained** (exp1111): AUROC=0.9946 on 7349-step PRM corpus (+0.6pp over v1).
   Model path confirmed and ready for use as reward signal in GRPO training.

7. **arXiv bundle complete** (exp1113): main.tex + carnot.bib + 7 figures, 100% citation
   resolution. pdflatex absent from conductor environment — compilation deferred. 2026-05-15
   deadline at risk. The exp1115 position paper v3 strategic rewrite further updated the paper
   (claims recalibrated: 13061x→15.6x, k=15→k_max=8, energy_inversion documented).

8. **LLM failure exemplar corpus v1** (exp1112): 36 exemplars, 12 categories, cascade TP
   rate 80.6%, mathematical-objective TP 100%, Goodfire-style exemplars TP 100%. Key weakness:
   hallucination_numeric and underspecification categories have only 33.3% TP — needs expansion.

9. **Energy inversion documented** (exp1115): mean_correct_energy=0.689 > mean_incorrect_energy=
   0.621 on SOTA outputs — documented as open methodological question in position paper v3. Not
   empirically resolved. Root cause per arXiv 2504.13134 (EBRM): OOD distribution shift from
   RL-optimized SOTA outputs vs base-model FoVer training corpus.

---

## Three Biggest Gaps to .87

### Gap 1: arXiv Submission Blocked — CRITICAL DEADLINE 2026-05-15

The position paper v3 is written (6044 words, 7 sections, empirical bounds framing) and the
LaTeX bundle is complete (exp1113). Only the pdflatex compilation step is missing. The 2026-05-15
deadline is 14 days from today. Risk: if not submitted by then, the paper misses the NeurIPS
2026 workshop window.

Fix: install texlive-full in conductor environment and compile. If that fails, submit the
.tex bundle directly to arXiv (arXiv accepts .tex source bundles and compiles them server-side).
Fallback: fill author identity placeholder in main.tex and tar+upload.

### Gap 2: Energy Inversion on SOTA Outputs — Phase-3 Credibility Risk

The energy ordering is INVERTED on SOTA model outputs: correct answers have HIGHER energy than
incorrect ones (mean_correct=0.689 > mean_incorrect=0.621 per exp1100/exp1115). This is
documented as an "open methodological question" in position paper v3, but it undermines the
core claim that "low energy = valid configuration."

Root cause per arXiv 2504.13134 (EBRM): The FoVer training corpus consists of base-model
GSM8K solutions (unoptimized, high error rate). SOTA model outputs (Qwen3.6-35B-A3B, Gemma4-31B)
are heavily RL-optimized and lie outside the FoVer training distribution. When the verifier
is evaluated OOD, its calibration inverts.

Fix: extend FoVer corpus with 1000+ SOTA model outputs (labeled by Z3+AST, not ThinkPRM to
avoid circularity), then retrain the energy verifier on the extended corpus. Measure whether
inversion is resolved.

### Gap 3: Self-Learning Architecture Is the Wrong Design

RLVR+SSD has given honest negatives for 3 consecutive milestones (exp1083, exp1099, exp1110).
The design flaw: RLVR treats energy as a binary reward signal (selected/not-selected), and SSD
distills from energy-selected examples using token KL loss. Neither mechanism uses ThinkPRM v2's
calibrated AUROC=0.9946 as a continuous reward signal.

Per arXiv 2509.21154 (GRPO is Secretly a PRM), GRPO training already implements an implicit
PRM via Monte Carlo completions over shared prefixes. Carnot's ThinkPRM v2 can replace the
implicit MC-sampled reward with a calibrated energy-based reward, giving higher signal quality.
The integration is simpler than RLVR+SSD: one GRPO training loop with energy rewards, no
separate SSD phase.

---

## Milestone Architecture

```
Phase 0: CRITICAL + Infrastructure (MANDATORY, unconditional)
         exp1116 arXiv Submission (deadline 2026-05-15)
         exp1117 Infrastructure Hardening v3 (manifest dispatch + doc async + bootstrap grace)

Phase 1: GRPO Energy PRM (continuous self-learning, GPU)
         exp1118 GRPO with ThinkPRM v2 as PRM reward
         [prior_failures: exp1110, exp1099, exp1083]

Phase 2: Energy Inversion Fix (SOTA corpus extension + retrain)
         exp1119 FoVer SOTA Extension v5 (GPU, generate 1000+ SOTA outputs)
              ↓
         exp1120 Energy Verifier Retrain + Inversion Fix
         [gated on exp1119.fover_sota_pairs_added_above_7000]
         [prior_failures: exp1100, exp1115]

Phase 3: k=5 Production Deployment
         exp1121 AND-Composition k=5 Production Wiring
         [standalone, no GPU]

Phase 4: FPGA v4 Python Simulation + Architecture Review
         exp1122 KV260 v4 Sparse+Inertia Python Sim + Hardware Attempt
         [prior_failures: exp1109, exp1094]

Phase 5: Adaptive Cascade Routing
         exp1123 Lagrangian Cascade Router (arXiv 2604.14853)
         [standalone, GPU optional]

Phase 6: WOPR Gallery
         exp1124 WOPR Hashi Puzzle Cartridge (codex)
              ↓
         exp1125 HF Spaces Gallery Update
         [gated on exp1124.hashi_cartridge_shipped]

Phase 7: Retro
         exp1126 Milestone 2026.04.87 Retrospective
```

---

## 11 Success Criteria

1. **arxiv_submitted_or_bundle_uploaded** (exp1116) — paper submitted to arXiv before 2026-05-15
2. **infrastructure_3_bottlenecks_fixed** (exp1117) — manifest dispatch + doc async + bootstrap grace
3. **grpo_energy_prm_honest_result** (exp1118) — GRPO training run completes with honest result
4. **fover_sota_pairs_above_7000** (exp1119) — FoVer corpus extended with SOTA outputs
5. **energy_inversion_measured_post_retrain** (exp1120) — retrained verifier energy ordering measured
6. **k5_and_compose_production_deployed** (exp1121) — VerifyRepairPipeline defaults to k=5 ensemble
7. **kv260_v4_kl_measured** (exp1122) — KL(v4_python_sim || Gibbs) measured
8. **adaptive_cascade_savings_measured** (exp1123) — Lagrangian router vs fixed cascade benchmarked
9. **hashi_cartridge_shipped** (exp1124) — WOPR Hashi cartridge with E=0 at convergence
10. **gallery_updated** (exp1125) — HF Spaces gallery updated with Hashi
11. **retro_complete** (exp1126) — milestone retrospective written

---

## Key Architectural Decisions for .87

1. **DualGPU MANDATORY for exp1118 GRPO** — streak broken in .86 (exp1110/1111); don't let
   it creep back. GRPO with N=8 completions per question benefits from parallel GPU inference.

2. **No gemini agent_type** — 429-rate-limited, useless for autonomous research. All
   long-context work via Opus.

3. **Codex for WOPR Hashi** — formulaic constraint encoding (Hashi bridge constraints are
   integer-flow + planarity, well-documented patterns; Codex excels here).

4. **FoVer SOTA extension is the prerequisite for energy inversion fix** — exp1119 gates
   exp1120. Do not re-run energy ordering measurement without the extended corpus.

5. **ThinkPRM NOT in k=5 AND-compose ensemble** — ThinkPRM×Z3Math r=0.507 at k=6;
   ThinkPRM stays as standalone Tier 0a. The k=5 ensemble is:
   [SOSKANEnergyV3, SemEnergyProbe, ASTStructureVerifier, SemanticConsistencyVerifier, Z3MathVerifier]

6. **GRPO replaces RLVR+SSD** — 3 consecutive honest negatives for RLVR+SSD. New approach:
   GRPO + ThinkPRM v2 rewards. Different training loop, different reward signal quality,
   different advantage estimation (group-relative cancels baseline drift).

7. **k=5 target is final for Phase-1d** — Phase-1d goal is k=5 AND-composition, not k=6
   (k=6 empirically blocked by ThinkPRM×Z3Math r=0.507). Update all specs and roadmap docs.

---

## Hardware Requirements

- **exp1118 GRPO**: DualGPU (2× RTX 3090) PREFERRED for N=8 parallel completions
- **exp1119 FoVer extension**: GPU (single RTX 3090 OK, DualGPU faster)
- **exp1120 Energy retrain**: GPU (single RTX 3090 OK)
- **exp1122 FPGA sim**: CPU only (Python simulation, no hardware synthesis)
- **exp1123 Cascade routing**: GPU optional (small MLP, trains on CPU in <5 min)

---

## Dependency Graph

```
exp1116 ─── (standalone, CRITICAL)
exp1117 ─── (standalone, CRITICAL)
exp1118 ─── (standalone, GPU)
exp1119 ─── (standalone, GPU)
            └──→ exp1120 (gated on exp1119.fover_sota_pairs_added_above_7000)
exp1121 ─── (standalone)
exp1122 ─── (standalone)
exp1123 ─── (standalone)
exp1124 ─── (standalone)
            └──→ exp1125 (gated on exp1124.hashi_cartridge_shipped)
exp1126 ─── (runs last, standalone)
```

---

## New arxiv Papers Incorporated

| Paper | ID | Action |
|---|---|---|
| EBRM: Energy-Based Reward Models | arXiv 2504.13134 | exp1120 training method |
| GRPO is Secretly a PRM | arXiv 2509.21154 | exp1118 training loop design |
| Lagrangian Cascade Routing | arXiv 2604.14853 | exp1123 cascade router |
| Dual BRAM Ising 800-node | arXiv 2602.16143 | exp1122 architecture reference |
| p-Bit with Inertia (parallel fix) | arXiv 2604.17109 | exp1122 v4 validation |
| Optimal Verification Granularity | arXiv 2505.11730 | exp1123 step vs response level |

---

## Position Paper Status

- **v3 written**: docs/position-paper-draft-v3.md (6044 words, 7 sections)
- **LaTeX bundle**: docs/arxiv-paper/ (main.tex + carnot.bib + 7 figures, 100% citations)
- **Blocking step**: pdflatex not installed; author identity placeholder needs filling
- **Target**: arXiv submission by 2026-05-15 (cs.LG primary)
- **exp1116 must close this**: install texlive or submit .tex bundle directly to arXiv

---

## What NOT to Propose in .87

- Gemini agent_type tasks (rate-limited, blocked since .84)
- k=6 AND-composition experiments (empirically blocked at r=0.507, Phase-1d = k=5)
- New RLVR+SSD experiments without a fundamentally different design (3 honest negatives)
- WOPR games that duplicate existing ones (Sudoku, GTW, Lights Out, N-Queens already live)
- Duplicate FoVer expansion without new model coverage (v5 adds SOTA outputs; v6 would
  need code/agentic domain expansion, not more GSM8K)
