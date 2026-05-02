# Research Roadmap — Milestone 2026.04.89

**Title:** Final Submission + Bounded Verification + CCTU Benchmark + Continuous Self-Learning Repair + Hardware Readiness

**CalVer:** 2026.04.89 (sequence increment from 2026.04.88)  
**Planned Experiments:** Exps 1139–1151 (13 experiments)  
**Date Designed:** 2026-05-02  
**Prerequisite:** Milestone 2026.04.88 retro complete (Exp 1138)

**Decentralization implications:** .89 preserves Carnot's local-first path: headline LLM work uses
mandated local GGUF models, closed APIs are not required for core results, and the Extropic/KV260
tasks are optional hardware-readiness paths rather than vendor lock-in.

---

## What Milestone 2026.04.88 Proved

Milestone .88 met 10 of 11 success criteria. It converted several .87 "promising but incomplete"
threads into stronger evidence, while surfacing three practical gaps that should define .89.

**Wins:**
- **k=5 AND-compose fixed:** SOSKANEnergyV3 root cause was non-convergence, not a simple sign
  error. Retraining/repair lifted SOS-KAN individual AUROC to 0.9902 and the k=5 ensemble AUROC
  to **0.9402**, clearing the >0.8 target.
- **GRPO + ThinkPRM v2 improved again:** 100-question training used DualGPU and produced
  **+8.51pp** improvement over baseline. Advantage standard deviation rose to 0.125, but remains
  below the desired >0.15 diversity target. Evaluation completed 47/50 holdout questions.
- **Zenil alpha_t improved:** post-retrain alpha_t reached **0.52** vs prior 0.38, strengthening
  the continuous self-learning signal.
- **Cascade v2 preserved accuracy:** Lagrangian cascade v2 achieved **3.2% cost savings** with
  **0.0pp accuracy delta** after adding verifier-score features.
- **Adversarial style probes did not fool k=5:** PRM-BiasBench-style attacks had **100% TP rate**
  under k=5; Z3 dominated style-irrelevant failures.
- **Position paper updated:** .88 findings were incorporated into the paper draft.

**Partials and failures:**
- **arXiv is still not fully submitted:** exp1127 compiled a PDF, but `arxiv_submitted=false`.
  Manual upload remains the highest operational priority before the 2026-05-15 deadline.
- **Goodfire-style exemplars are caught late:** k=5 catches all 36 curated exemplars, but cheap
  tiers remain weak: ThinkPRM 13.9%, SemEnergy 22.2%, SymCode 8.3%, Causal 2.8%, Z3 standalone
  8.3%. The cascade is accurate because it escalates, not because early tiers are calibrated.
- **KV260 sampler remains above target:** v4 tuning improved KL to 0.1128, still above the
  <0.05 threshold. Self-adaptive lambda worsened the mismatch, suggesting a topology/update-rule
  issue rather than a scalar-parameter issue.
- **Slitherlink did not ship:** exp1136 was blocked by a stale gate/prior-failures path, and the
  HF gallery update was consequently absent.
- **Roadmap hygiene still leaks time:** stale `prior_failures` and gate metadata caused false
  block/waste patterns despite the conductor itself being stable.

---

## Three Biggest Gaps Between Current State and PRD Vision

1. **Verified reasoning still lacks distribution-level certificates.**  
   k=5 is now strong on sampled outputs, but Carnot does not yet bound the probability mass of
   invalid outputs under a model distribution. The PRD's verifiable-reasoning vision needs a
   certificate tier: "under these constraints, unsafe mass is <= p." BEAVER-style deterministic
   bounds are the most direct next step.

2. **The cascade is too dependent on expensive late tiers.**  
   .88 proved k=5 can catch curated failures, but early tiers are poorly calibrated for Goodfire
   exemplar classes. Practical local-first verification needs cheap routing that knows when to
   escalate, especially for data-driven vs reasoning-driven hallucination modes.

3. **Continuous self-learning and hardware are not yet closed loops.**  
   GRPO and alpha_t improved, but the system is not yet using its own repair attempts as a
   persistent self-learning signal. Hardware is also still a simulation story: KV260 KL remains
   above target, and Extropic/thermodynamic paths need concrete integration packets before the
   architecture can claim hardware readiness.

---

## External Findings Incorporated

The 2026-05-02 literature and ecosystem scan added the following to `research-references.md` before
this roadmap was designed:

- **BEAVER (arXiv 2512.05439):** deterministic probability bounds for semantic constraints.
- **HalluGuard (arXiv 2601.18753):** data-driven vs reasoning-driven hallucination decomposition.
- **CCTU (arXiv 2603.15309):** constrained tool-use benchmark with executable validators.
- **RandCSPBench (arXiv 2602.18419):** hard-regime CSP benchmark to avoid easy-instance wins.
- **HardNet++ (arXiv 2604.19669) and KKT-Hardnet (arXiv 2507.08124):** differentiable hard
  constraint projection for repair.
- **EBT-Policy (arXiv 2510.27545):** adaptive Langevin dynamics for behavior-space EBMs.
- **MetaCluster KAN compression (arXiv 2510.19105):** prototype-clustered KAN compression.
- **Energy-Time-Accuracy thermodynamic computing (arXiv 2601.04358):** hardware benchmark framing.
- **Extropic XTR-0/Z1 updates:** TSU hardware path and THRML software integration.
- **Logical Intelligence Kona/Aleph updates:** neural theorem-proving target for future bridges.
- **MARCH (arXiv 2603.24579):** multi-agent information-asymmetric self-checking.
- **NRGPT (arXiv 2512.16762) and transformer intrinsic-optimizer framing (arXiv 2511.00907):**
  useful Phase 3 seeds for energy-native autoregressive inference and accelerated energy updates,
  but not .89 critical path.
- **DiffuTruth / Energy of Falsehood (arXiv 2602.11364):** candidate factuality feature for
  future Goodfire cheap-tier work if HalluGuard features are insufficient.
- **MCP Solver and PyCSP3 repositories:** practical solver/backend patterns for the CCTU adapter
  and future CSP benchmark expansion.

---

## Architecture Diagram

```
                 Carnot .89 Target Architecture
                 ==============================

Inputs: math, code, tool-use, curated hallucination exemplars, hard CSPs
    │
    ▼
SOTA Local Generation
    ├─ unsloth/Qwen3.6-35B-A3B-GGUF
    ├─ unsloth/gemma-4-31B-it-GGUF
    └─ unsloth/gemma-4-26B-A4B-it-GGUF
    │
    ▼
Cheap Tiers + Router v3
    ├─ ThinkPRM v2
    ├─ SemEnergy
    ├─ HalluGuard-style NTK/data-vs-reasoning features
    └─ Goodfire exemplar calibration
    │           │
    │           └── escalate when cheap tiers are low-confidence
    ▼
k=5 AND-compose verifier (AUROC 0.9402 from .88)
    ├─ SOSKANEnergyV3
    ├─ SemEnergyProbe
    ├─ ASTStructureVerifier
    ├─ SemanticConsistencyVerifier
    └─ Z3MathVerifier
    │
    ├── BEAVER-lite certificate tier
    │       └─ bound unsafe probability mass for prefix-closed constraints
    │
    ├── Executable benchmark tier
    │       └─ CCTU 25-task adapter for tool-use constraints
    │
    └── Repair + self-learning loop
            ├─ GRPO reflection reward: r = E_before - E_after
            ├─ HardNet++/KKT projection repair for numeric constraints
            └─ persistent repair trace corpus for FR-11

Hardware readiness path
    ├─ KV260 v5 DC-continuous Ising diagnostic (software parity first)
    └─ Extropic Z1/XTR-0 integration packet + ETA benchmark spec
```

---

## Phase Descriptions

### Phase 0 — Release Blockers and Roadmap Hygiene

**exp1139: arXiv Final Submission Close-Out**

The position paper is no longer a writing problem; it is an operational submission problem. This
task must use the exp1127 PDF/artifacts, verify the latest PDF and source bundle, and either record
the arXiv submission ID or produce the exact remaining manual actions with no ambiguity.

Acceptance: `arxiv_submitted=true` with `arxiv_id`, or a checked artifact proving
`pdf_compiled=true` and `manual_upload_steps_remaining` is precise enough to complete immediately.

**exp1140: Roadmap Gate/Prior-Failures Audit v1**

.88 lost Slitherlink and wasted agent calls on stale gate/prior-failure metadata. Because the
conductor itself is not to be modified in .89 planning, this experiment builds an external audit
script and report that checks the new roadmap for structured gates, stale upstream IDs, missing
prior-failure declarations on carry-forwards, and unsupported model/agent combinations.

Acceptance: `roadmap_gate_audit_passed=true`, with a report covering all .89 tasks.

**exp1141: WOPR Slitherlink Rescue**

Slitherlink is a formulaic constraint cartridge, and .88 did not fail on the modeling problem. The
rescue should explicitly declare the exp1136 blocked-gate failure and implement the cartridge with
hard-regime test instances informed by RandCSPBench.

Acceptance: `slitherlink_cartridge_shipped=true`, `e_zero_at_convergence=true`, with tests.

### Phase 1 — Bounded Verification and Benchmark Breadth

**exp1142: BEAVER-Lite Deterministic Bounder**

Carnot needs a certificate tier above sampled verification. This task implements a local, scoped
BEAVER-lite bounder for prefix-closed arithmetic constraints using SOTA local GGUF models, compares
bound tightness to empirical sampling, and records whether token-logprob access is sufficient.

Acceptance: `beaver_lite_bounder_written=true`, `unsafe_mass_bound_reported=true`.

**exp1143: HalluGuard Cascade Router v3**

The router should learn why cheap tiers fail, not just that they fail. Add HalluGuard-style features
for data-driven mismatch vs reasoning-driven decoding instability and test whether these explain
Goodfire exemplar routing.

Acceptance: `halluguard_features_added=true`, with cost savings and TP-rate deltas measured.

**exp1144: CCTU 25-Task Local SOTA Adapter**

CCTU is the right next FR-12 benchmark: agentic tool-use under explicit constraints with executable
validation. This task builds a 25-task local adapter and runs at least one mandated SOTA GGUF model.

Acceptance: `cctu_adapter_written=true`, `cctu_tasks_evaluated>=25`, and executable validation
rates reported.

**exp1145: Goodfire Cheap-Tier Distillation v1**

After exp1143 identifies the weak-tier failure classes, this task distills the curated Goodfire
exemplar signal into cheap-tier thresholds/features without weakening k=5 safety.

Acceptance: `cheap_tier_tp_rate_improved=true`, with SemEnergy/ThinkPRM cheap-tier TP rate and
false-positive deltas reported.

### Phase 2 — Continuous Self-Learning and Repair

**exp1146: GRPO Reflection Reward v3**

This is the mandatory continuous self-learning experiment. It adds a repair-grounded reward:
`r_reflect = E_before - E_after`, using the model's own failed and repaired outputs as persistent
training signal. It must use mandated SOTA local GGUF models and DualGPU.

Acceptance: `reflection_reward_integrated=true`, `improvement_over_baseline>=0.09` or an honest
negative with advantage-diversity diagnostics.

**exp1147: HardNet++ / KKT Projection Repair**

Prompt-based repair is expensive and unstable for numeric constraints. This task prototypes a
projection layer for continuous arithmetic/range constraints and compares it to current prompt
repair.

Acceptance: `projection_repair_written=true`, with violation-rate and latency comparisons.

**exp1148: MetaCluster SOS-KAN Compression**

SOS-KAN is now strong but may be too heavy for cheap-tier use and hardware paths. This task tests
whether MetaCluster-style prototype compression preserves AUROC while shrinking the model.

Acceptance: `sos_kan_compressed=true`, `auroc_drop<=0.02`, and `size_reduction_factor>=5`.

### Phase 3 — Hardware Readiness and Retro

**exp1149: KV260 v5 DC-Continuous Ising Diagnostic**

.88 tuning suggests the mismatch is structural. This task prototypes a DC-continuous Ising software
fallback, compares it to v4, and frames results with energy-time-accuracy diagnostics.

Acceptance: `kv260_v5_diagnostic_complete=true`, with KL and ETA metrics reported.

**exp1150: Extropic Z1/XTR-0 Integration Packet**

Carnot needs a concrete hardware integration packet before vendor hardware arrives. This task
produces a THRML parity benchmark spec, minimal EBM workload suite, and early-access checklist for
Extropic Z1/XTR-0.

Acceptance: `extropic_integration_packet_written=true`, with benchmark workload files.

**exp1151: Milestone 2026.04.89 Retrospective**

Evaluate all success criteria, gate outcomes, runtime, SOTA-model compliance, and whether the
release, certification, self-learning, and hardware-readiness questions advanced.

Acceptance: `retro_complete=true`.

---

## Dependency Graph

```
Phase 0:
  exp1139  arXiv final submission close-out
  exp1140  roadmap gate/prior-failures audit
  exp1141  Slitherlink rescue

Phase 1:
  exp1142  BEAVER-lite certificate tier
  exp1143  HalluGuard router v3
      └── exp1145 Goodfire cheap-tier distillation
  exp1144  CCTU 25-task adapter

Phase 2:
  exp1146  GRPO reflection reward continuous self-learning
  exp1147  HardNet++/KKT projection repair
  exp1148  MetaCluster SOS-KAN compression

Phase 3:
  exp1149  KV260 v5 DC-continuous Ising diagnostic
  exp1150  Extropic Z1/XTR-0 integration packet
  exp1151  milestone retro
```

Structured conductor gates:
- exp1145 is gated on `exp1143.halluguard_features_added == true`.
- exp1151 is intentionally ungated so the milestone always produces a retro, even if upstream tasks fail.

---

## Hardware Requirements

| Area | Tasks | Required Hardware | Notes |
|------|-------|-------------------|-------|
| SOTA local LLM inference | exp1142, exp1144, exp1146 | 2x RTX 3090 preferred | Use mandated GGUF models; small legacy models only for CPU smoke tests. |
| DualGPU training | exp1146 | 2x RTX 3090 | Mandatory for GRPO reflection reward. |
| CPU-only diagnostics | exp1140, exp1141, exp1143, exp1145, exp1147, exp1148, exp1151 | Workstation CPU | Avoid live model calls unless explicitly required. |
| KV260 path | exp1149 | No live KV260 required | Software diagnostic first; Vivado still not assumed installed. |
| Extropic path | exp1150 | No hardware required | Produces integration packet for future Z1/XTR-0 access. |

---

## Success Criteria

1. `arxiv_submitted_or_manual_packet_finalized` (exp1139) — CRITICAL before 2026-05-15.
2. `roadmap_gate_audit_passed` (exp1140) — no stale gates or missing carry-forward prior_failures.
3. `slitherlink_cartridge_shipped` (exp1141) — E=0 at convergence with tests.
4. `beaver_lite_bound_reported` (exp1142) — unsafe-mass bound and empirical comparison written.
5. `halluguard_router_features_measured` (exp1143) — data-vs-reasoning features evaluated.
6. `cctu_adapter_honest_result` (exp1144) — 25 constrained tool-use tasks evaluated.
7. `goodfire_cheap_tier_tp_improved_or_honest_negative` (exp1145).
8. `grpo_reflection_reward_honest_result` (exp1146) — continuous self-learning requirement.
9. `projection_repair_honest_result` (exp1147).
10. `sos_kan_compression_honest_result` (exp1148).
11. `kv260_v5_diagnostic_honest_result` (exp1149).
12. `extropic_integration_packet_written` (exp1150).
13. `retro_complete` (exp1151).

Passing threshold: 10 of 13 success criteria, with exp1139, exp1146, and exp1151 treated as
mandatory strategic criteria.

---

## Key Decisions for .89

- **Run release close-out first.** The arXiv deadline is calendar-bound and must not be buried under
  research tasks.
- **Do not modify `scripts/research_conductor.py` in .89 tasks.** Roadmap/gate hygiene is handled
  by an external audit script/report.
- **Use mandated SOTA local GGUF models for every LLM experiment.** Legacy small models are allowed
  only as CPU smoke tests.
- **No Gemini routing.** Gemini remains unsuitable for scheduled milestone work due to recent 429
  pauses; Codex is used only for formulaic Slitherlink code.
- **Treat k=5 as a strong verifier but not a certificate.** BEAVER-lite is introduced to begin
  distribution-level guarantees.
- **Treat KV260 KL as a structural diagnostic problem.** .89 should test an alternate continuous/DC
  formulation instead of another scalar sweep.

---

## Estimated Wall Time

| Phase | Experiments | Estimate |
|-------|-------------|----------|
| Phase 0 release/hygiene | exp1139-exp1141 | 90 min |
| Phase 1 verification breadth | exp1142-exp1145 | 200 min |
| Phase 2 self-learning/repair | exp1146-exp1148 | 240 min |
| Phase 3 hardware/retro | exp1149-exp1151 | 190 min |
| **Total** | **13 experiments** | **~720 min** |

The estimate is intentionally higher than .88 because .89 includes three SOTA-local model tasks, one
DualGPU self-learning run, and two hardware-readiness tasks. The structured gates and audit task are
expected to avoid the specific stale-gate failure class that blocked Slitherlink in .88.
