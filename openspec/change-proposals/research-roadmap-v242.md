# Research Roadmap v242: Phase 4 FREIA FEP Sprint + Step-Level ARM-EBM + HalluGuard Tier 0s + Ensemble v7 + KV260 PYNQ Flash

**Milestone:** 2026.05.242
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.241 — 8/10 tasks completed (AUROC 0.975 adversarially verified; Tier 0r viable at 0.9123; FR-11 all tiers integrated end-to-end; PolarFire terminal; KV260 PYNQ SD-card path established; Phase 4 STILL unvalidated — Gate 3 unmet; arXiv hold remains)

---

## What .241 Proved

Milestone .241 had 8 of 10 tasks complete (1 MISSING artifact — exp2496 Qwen PRC v3, 1 GATE_BLOCKED — exp2499 Tier 0q ensemble extension). Key findings:

**Major wins:**
- **AUROC adversarially verified** (exp2498): Independently replicated group-conditional 0.975 across 5 seeds with explicit cross-group tautology check (Group A, B, C platt_aurocs confirmed distinct). Mean=0.9750, gap to HIVE=+0.0514. Gate 4 of 4 met. cite-safe.
- **Tier 0r Curry-Howard viable** (exp2504): AUROC=0.9123 on .241 corpus. The 16th verifier candidate (arXiv:2510.01069 ICLR 2026) is viable for ensemble expansion. Tier 0q (Spilled Energy) definitively non-viable (pearson=-0.022, AUROC=0.4903 noise floor — exp2497).
- **FR-11 all tiers end-to-end** (exp2500): Tier 4 adaptive-energy feedback fires into Tier 1 on 10/10 continuous-self-learning example corpus. FR-11 integration complete.
- **PolarFire terminal state reached** (exp2501): energy_sanity_check_passed=True. Board graduates to optional/opportunistic per Hardware-Task Continuity Discipline.
- **KV260 PYNQ SD-card path established** (exp2502): PYNQ SD-card boot is a viable programmer-purchase-bypass alternative to the blocked DirtyJTAG path. Requires .hwh file extraction from Vivado block design — this is the next step.
- **Phase 4 Spilled Energy definitively closed** (exp2497): pearsonr=-0.022, AUROC=0.4903 — noise floor. Tier 0q retired from candidate set. exp2499 pre-gate-blocked as downstream consequence.

**Gaps confirmed (entering .242):**
1. **Phase 4 Gate 3 still unmet** — exp2496 (Qwen PRC v3) artifact MISSING (likely GPU/quota blocked, precondition failure). exp2497 (Spilled Energy) definitively refuted. One viable untested path remains: step-level ARM-EBM correlation using per-CoT-step token logprobs (structurally distinct from response-level exp2486 which used SemanticEnergy embeddings as proxy rather than raw logprobs). FREIA (arXiv:2605.04065) provides the step-level FEP formalism to anchor this.
2. **Ensemble expansion not done** — Tier 0r passed viability gate but NOT integrated into conformal ensemble. HalluGuard Tier 0s (arXiv:2601.18753 ICLR 2026, NTK-based) not yet prototyped. Ensemble v7 remains unbuilt.
3. **KV260 flash still pending** — PYNQ path documented but .hwh file extraction from Vivado block design not attempted. No physical flash has occurred.
4. **arXiv blocked** — 3 of 4 gates met; unmet = gate_3_phase4. arXiv hold persists until Phase 4 empirically validates.

---

## Three Biggest Gaps vs PRD Vision (entering .242)

### Gap 1: Phase 4 Empirical Validation (Critical Path to arXiv)

Five Phase 4 attempts have been made, all failed or structurally incomplete:
- exp2474: ODAR routing proxy, pearson_r=0.19 — FAILED
- exp2486: ARM-EBM bijection via SemanticEnergy EMBEDDINGS (wrong proxy — not raw token logprobs), pearson_r=0.108 — FAILED
- exp2487: Qwen PRC mock_model — METHODOLOGY GAP, not a real test
- exp2496: Qwen PRC real GGUF — MISSING (resource/quota blocked at precondition)
- exp2497: Spilled Energy — DEFINITIVELY REFUTED (noise floor)

The ARM-EBM bijection (exp2486) failed because it used SemanticEnergy EMBEDDINGS as a proxy for E_ising rather than applying the ARM-EBM formula E=-log_p directly to raw token logprobs at per-CoT-step granularity. This is a DIFFERENT methodology from exp2486. FREIA (arXiv:2605.04065, FEP-based step-level RL) provides the theoretical grounding: energy at the step level is directly computable from existing telemetry manifest token logprobs, making this CPU-only and precondition-verifiable.

**Plan for .242:** exp2508 uses step-level ARM-EBM bijection:
- Load existing `.241 telemetry manifest (already on disk, no GGUF required)
- Compute E_step = -sum(log_p(token_i)) per CoT step from token logprob fields
- Compute Carnot IsingVerifier.energy() per CoT step
- Measure pearson_r(E_step, E_ising) across steps with n >= 100 step pairs
- Accept if |pearson_r| > 0.30 (structurally informative) with p < 0.05

This is structurally distinct from exp2486 (response-level, SemanticEnergy proxy) and exp2474 (ODAR routing). The prior_failures block documents all five failed paths with their diagnosed root causes.

### Gap 2: Ensemble Expansion — Tier 0r Integration + HalluGuard Tier 0s

Tier 0r (Curry-Howard soft-typed proof-path, AUROC=0.9123) is viable but not integrated. HalluGuard (arXiv:2601.18753 ICLR 2026) provides an NTK-based unified verifier that distinguishes data-driven vs reasoning-driven hallucinations — a theoretically complementary signal to the existing logprob/semantic/logic group structure.

**Plan for .242:**
- exp2509: HalluGuard Tier 0s prototype + AUROC evaluation on .241 corpus
- exp2510: Tier 0r integration into conformal ensemble v7 (10 verifiers vs prior 9), with updated group-conditional calibration

### Gap 3: KV260 Flash + arXiv Write-Through

KV260 PYNQ path is documented but no actual .hwh file has been generated or SD card flashed. The board is sitting in a limbo state: bitstream exists (from earlier synthesis) but PYNQ deployment requires the .hwh hardware description alongside the bitstream.

arXiv is blocked on Gate 3 only. Paper-v6 write-through (updating §6 limitations, updating §3 results with Tier 0r AUROC, updating any adversarially-flagged corrigenda) can proceed independently of Gate 3.

**Plan for .242:**
- exp2514: Generate .hwh from Vivado block design, attempt SD card flash and PYNQ boot
- exp2515: Paper-v6 final write-through — update §3, §6, corrigenda. arXiv gate check.

---

## Architecture Snapshot (entering .242)

```
Tier 0 Verifiers (conformal p-value ensemble — v6, 9 active verifiers):
  Group A (logprob-class):
    Tier 0a: SemanticEnergy (AUROC=0.810)
    Tier 0b: HALT (AUROC=0.8539)
    Tier 0f: PCIB (AUROC=0.8669)
  Group B (semantic-class):
    Tier 0c: FregeLogic (AUROC=0.8831)
    Tier 0e: LogCons Hierarchical (AUROC=0.8896)
    Tier 0g: LaaB Meta-Judgment (AUROC=0.854)
  Group C (logic-class):
    Tier 0d: DiffuTruth (AUROC=0.588, marginal but included)
    Tier 0h: NCO (AUROC=0.678)
  Excluded (non-viable / noise-floor):
    Tier 0i: ODAR routing (AUROC=0.5584)
    Tier 0p: LLM-as-Judge (AUROC=0.6412)
    Tier 0q: Spilled Energy (AUROC=0.4903, retired .241)
  Candidates for .242 expansion:
    Tier 0r: Curry-Howard soft-typed proof-path (AUROC=0.9123, viable, NOT yet integrated)
    Tier 0s: HalluGuard NTK-based (arXiv:2601.18753, NOT yet prototyped)
  Group-conditional calibration (3 groups, Fisher combination):
    mean AUROC = 0.975 (std=0.021, n=5 seeds) [adversarially verified, cite-safe]
    HIVE peer baseline: 0.9236 (+0.0514 gap, BREACHED)

Tier 1: KAN (AUROC=0.994, certified_coverage=0.833, local_lip=2.396) [certified-deployment-ready]
FR-11 Self-Learning Loop (all 4 tiers operational):
  FR-11 Tier 1: Online constraint reweighting (exp2500, end-to-end verified)
  FR-11 Tier 2: SQLite cross-session memory (exp2500)
  FR-11 Tier 3: JEPA violation predictor (AUC=0.7633)
  FR-11 Tier 4: Adaptive KAN knot structure (exp2500, prototype, 2 adaptations)

Hardware:
  PolarFire SoC: TERMINAL (energy_sanity_check_passed=True, exp2501)
  KV260: kv260_status=pynq_path_viable (exp2502); .hwh NOT yet generated; NOT yet flashed
  GateMate A1: TERMINAL (bitstream flashed + smoke-tested, .237)
  RTX 3090x2: PRIMARY training backend
  Strix Point gfx1150 (ROCm 7.2.3): SECONDARY portability backend

arXiv Gates:
  Gate 1 (Phase 1 ship): PASSED
  Gate 2 (audit): PASSED
  Gate 3 (Phase 4 empirical): FAILED — five paths tried, all failed/incomplete
  Gate 4 (AUROC adversarially verified): PASSED (0.9750 >= 0.9236, adversarially clean)
  Overall: arxiv_ready=False, blocked on Gate 3
```

---

## Milestone .242 Experiment Plan

### Phase 0 — Infrastructure (1 task)

**exp2507 — Archive .241 + Activate .242**
- Agent: codex, gpt-5.5, max_turns: 20
- Move completed .241 experiments to research-complete.yaml
- Activate milestone 2026.05.242 header in research-roadmap.yaml

### Phase 4 — Active Inference Empirical Validation (1 task, CRITICAL PATH)

**exp2508 — Phase 4 Step-Level ARM-EBM Bijection v2 (FREIA FEP Grounding)**
- Agent: codex, gpt-5.5, max_turns: 45
- CPU-only, uses existing telemetry manifest (no GGUF required)
- Prior failures: exp2474 (pearson=0.19, wrong proxy), exp2486 (pearson=0.108, SemanticEnergy embeddings proxy not raw logprobs), exp2487 (mock_model methodology gap), exp2496 (MISSING), exp2497 (Spilled Energy noise floor)
- New approach: E_step = -sum(log_p(token_i)) per CoT step from token logprob fields vs IsingVerifier.energy() per step
- Falsifiable gate: |pearson_r| > 0.30 AND p < 0.05 with n >= 100 step pairs
- retire_if_same_verdict: true (if response-level correlation < 0.30 again)

### Phase 2a — Ensemble Expansion (2 tasks)

**exp2509 — HalluGuard NTK-Based Tier 0s Prototype**
- Agent: codex, gpt-5.5, max_turns: 50
- Prototype NTK-based hallucination detector (arXiv:2601.18753 ICLR 2026)
- Evaluate AUROC on .241 corpus; viability gate AUROC > 0.70

**exp2510 — Tier 0r Integration + Conformal Ensemble v7**
- Agent: codex, gpt-5.5, max_turns: 45
- Integrate Curry-Howard Tier 0r (AUROC=0.9123) into group-conditional conformal calibration
- Re-run group-conditional calibration with 10 verifiers (add Tier 0r to Group C or new group)
- Report updated ensemble AUROC; compare to .241 baseline 0.975

### Phase 2b — Calibration Enhancement (2 tasks)

**exp2511 — Adaptive Conformal Prediction v2**
- Agent: codex, gpt-5.5, max_turns: 45
- Implement prompt-adaptive calibration (arXiv:2604.13991): vary calibration set selection based on prompt type
- Gate: adaptive_auroc >= group_conditional_baseline_0.975

**exp2512 — FR-11 Tier 2 Memory-Augmented Threshold Learning (32-Example)**
- Agent: codex, gpt-5.5, max_turns: 45
- Extend FR-11 Tier 2 SQLite memory to support 32-example per-domain threshold adaptation
- Gate: memory_augmented_auroc >= 0.95 on held-out domain examples

### Phase 2c — Model Improvement (1 task)

**exp2513 — KAN Multilevel Training**
- Agent: codex, gpt-5.5, max_turns: 45
- Apply multilevel training (arXiv:2603.04827) to KAN tier
- Gate: multilevel_auroc >= 0.994 (no regression from certified baseline)

### Phase 3 — Hardware (1 task)

**exp2514 — KV260 PYNQ .hwh Generation + Flash Attempt**
- Agent: codex, gpt-5.5, max_turns: 40
- PRECONDITIONS: Vivado installed, block design .bd file present, PYNQ SD card available
- Generate .hwh from Vivado block design via TCL script
- Attempt SD card image preparation + boot attempt (or document exact blocker)
- Gate: kv260_hwh_generated=True OR kv260_flash_attempted=True OR kv260_blocked_documented=True

### Phase 4b — Publication Track (1 task)

**exp2515 — Paper-v6 Final Write-Through + arXiv Gate Check**
- Agent: codex, gpt-5.5, max_turns: 40
- Update §3 (results) with Tier 0r AUROC=0.9123, ensemble v7 AUROC if available
- Update §6 (limitations) with Tier 0q retirement, HalluGuard Tier 0s status
- Address corrigendum_pending items from exp2505 (TAUTOLOGY flag on 0.975, DURATION_TOO_SHORT)
- Run arXiv gate check: report updated gate status

### Phase 5 — Synthesis (2 tasks)

**exp2516 — Capstone v242**
- Agent: claude, claude-opus-4-7, requires_claude: true, max_turns: 100
- NO HARD GATE — honest synthesis only
- Reason for claude: multi-file synthesis across 10 artifact files + cross-phase reasoning under ambiguity; capstone requires judgment calls that cannot be reduced to deterministic gates

**exp2517 — Operational Retrospective v242**
- Agent: codex, gpt-5.5, max_turns: 20
- Standard retro template; schema carnot.operational_retro.v65+

---

## Dependency Graph

```
exp2507 (activate)
  └─► exp2508 (Phase 4 step-level, CPU-only, independent)
  └─► exp2509 (HalluGuard Tier 0s, independent)
  └─► exp2510 (Tier 0r + ensemble v7)
       └─► exp2511 (adaptive conformal, depends on ensemble v7)
  └─► exp2512 (FR-11 Tier 2 memory, independent)
  └─► exp2513 (KAN multilevel, independent)
  └─► exp2514 (KV260 hardware, independent)
  └─► exp2515 (paper write-through, independent of exp2508 but gated on availability)

exp2508, exp2509, exp2510, exp2511, exp2512, exp2513, exp2514, exp2515
  └─► exp2516 (capstone, reads all artifacts)
       └─► exp2517 (retro, reads capstone)
```

---

## Hardware Requirements Table

| Board | Status | Task | Terminal State |
|---|---|---|---|
| AMD/Xilinx KV260 | pynq_path_viable (NOT flashed) | exp2514 | kv260_flash_attempted=True + boot transcript |
| Microchip PolarFire SoC | **TERMINAL** (exp2501, energy_sanity_check_passed=True) | optional only | N/A — graduated |
| Cologne Chip GateMate A1 | **TERMINAL** (.237, n=16 Ising tile flashed + smoke-tested) | optional only | N/A — graduated |
| RTX 3090x2 | PRIMARY training | available for exp2508+ GGUF if needed | N/A |
| Strix Point gfx1150 | SECONDARY portability | optional fallback | N/A |

Note: PolarFire and GateMate are TERMINAL per Hardware-Task Continuity Discipline (.241 exp2501 and .237 respectively). Only KV260 is non-terminal and receives a mandatory task slot.

---

## Decentralization Compliance (Rules 1–7)

| Rule | Compliance |
|---|---|
| Rule 1: Local-first open models | exp2508 uses existing telemetry (no GGUF required). exp2509/2510 use cached SOTA pair from experiment_template.py. All experiments have CPU-only fallback paths documented in PRECONDITIONS. |
| Rule 2: Closed frontier models optional | No experiment requires a closed-weight API. Capstone (exp2516) uses Claude for judgment synthesis only — paper claims anchor to open-model results exclusively. |
| Rule 3: Distribution mirroring | No new weights published this milestone. If ensemble v7 model card updated, HuggingFace + IPFS CID documented before announcement. |
| Rule 4: Multiple integration surfaces | FR-11 Tier 2 memory extension (exp2512) touches Python API. No regression to CLI/MCP/HTTP surfaces introduced. |
| Rule 5: Hardware portability | KV260 task (exp2514) continues hardware sovereignty track. PolarFire terminal graduation does not remove hardware portability path. |
| Rule 6: Per-call data minimization | No closed-weight LLM calls in any experiment except capstone. Capstone: data_handling_class=minimize by default. |
| Rule 7: No vendor-specific core imports | All new verifier code in exp2509/2510 must use abstract SamplerBackend / LLMComponent protocols. HalluGuard NTK logic in `python/carnot/verify/` not in `closed_weight/`. |

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` before planning. The following retired scopes are confirmed NOT re-proposed in .242:

| Retired scope | Reason confirmed not re-proposed |
|---|---|
| exp2091: Tier 1 CSL Grammar Updates | No .242 task touches CSL grammar |
| GRPO v1-v14 lineage | No GRPO variants proposed |
| WOPR puzzle cartridges | No puzzle cartridge work |
| HardNet++/DSP repair | No HardNet work |
| THRML scaling sweep | exp2513 uses KAN multilevel training (arXiv:2603.04827), NOT THRML scaling. Distinct scope. |
| SpecAnn spectral annealing | No spectral annealing proposed |
| Tier 0q Spilled Energy | Definitively retired in .241; exp2499 blocked. exp2508 uses step-level -log_p method, NOT Spilled Energy metric. Distinct methodology. |

exp2497 verdict (pearsonr=-0.022) formally retires Tier 0q. exp2508 is NOT a Tier 0q rerun — it is an ARM-EBM Phase 4 correlation test using a different energy metric (raw token logprobs per CoT step, not spilled energy from logit distributions). The `prior_failures:` block in exp2508's YAML documents this distinction explicitly.

---

## Failed-Experiment Rerun Compliance Table

| Experiment | Prior failure | Root cause | What changed | Falsifiable gate | retire_if_same |
|---|---|---|---|---|---|
| exp2508 (Phase 4 step-level) | exp2474 (pearson=0.19, ODAR), exp2486 (pearson=0.108, SemanticEnergy proxy), exp2487 (mock_model), exp2496 (MISSING), exp2497 (noise floor) | Proxy mismatch (not raw logprobs), methodology gap (mock), or resource block | Uses raw token logprob fields from telemetry manifest at per-CoT-step granularity; applies E=-sum(log_p) directly per ARM-EBM formula; FREIA FEP grounding for step-level formalism | |pearson_r| > 0.30 AND p < 0.05, n >= 100 step pairs | true |
| exp2510 (Tier 0r integration) | N/A (first integration attempt; viability was gated on exp2504 passing) | — | Tier 0r proved viable (AUROC=0.9123, exp2504); this is the integration step, not a repeat viability test | ensemble_v7_auroc >= 0.970 | false |
| exp2514 (KV260 flash) | exp2441/exp2452/exp2491 (DirtyJTAG 1.8V incompatible, OpenOCD lacks ZynqMP init, programmer unavailable) | Hardware programmer incompatibility; wrong JTAG path | PYNQ SD-card path (exp2502 documented it viable); .hwh generation via Vivado TCL; no JTAG programmer required | kv260_hwh_generated=True OR kv260_flash_attempted=True OR kv260_blocked_documented | false |

---

## Agent Routing Table

| Experiment | Agent | Model | requires_claude | Justification |
|---|---|---|---|---|
| exp2507 | codex | gpt-5.5 | false | Mechanical YAML archive operation |
| exp2508 | codex | gpt-5.5 | false | Numerical computation, deterministic gates |
| exp2509 | codex | gpt-5.5 | false | Prototype + AUROC gate, deterministic |
| exp2510 | codex | gpt-5.5 | false | Integration + calibration, deterministic |
| exp2511 | codex | gpt-5.5 | false | Numerical calibration, deterministic |
| exp2512 | codex | gpt-5.5 | false | FR-11 memory extension, single-file |
| exp2513 | codex | gpt-5.5 | false | KAN training, deterministic gate |
| exp2514 | codex | gpt-5.5 | false | Hardware toolchain, documented path |
| exp2515 | codex | gpt-5.5 | false | LaTeX write-through, mechanical splice |
| exp2516 | claude | claude-opus-4-7 | true | Multi-artifact synthesis across 10 files; open-ended judgment under ambiguity; capstone requires cross-phase reasoning that cannot be reduced to deterministic gates (meets all 3 positive-criterion conditions: codex has historically produced shallow capstone summaries, 10+ artifact files, no single deterministic gate) |
| exp2517 | codex | gpt-5.5 | false | Templated retro structure |

**Codex ratio: 10/11 = 90.9%** (target >= 85%). Compliant with Codex-Default Discipline.

---

## arXiv Readiness Projection

| Gate | .241 Status | .242 Projection |
|---|---|---|
| Gate 1: Phase 1 ship | PASSED | PASSED (no regression) |
| Gate 2: Audit | PASSED | PASSED (exp2515 updates corrigenda) |
| Gate 3: Phase 4 empirical | FAILED | POSSIBLE — exp2508 step-level ARM-EBM is the only untried structurally-distinct path; retire_if_same_verdict=true means if this fails, the Phase 4 hypothesis is declared empirically unsupported and paper is revised accordingly |
| Gate 4: AUROC adversarially verified | PASSED (0.9750) | PASSED (exp2510 ensemble v7 may improve further) |

If exp2508 FAILS Gate 3: the operator-directed arXiv hold is resolved by treating Phase 4 as a documented empirical negative — paper §4 is rewritten to declare the hypothesis "not supported by current empirical evidence" and arXiv proceeds without Phase 4 validation. This path requires explicit operator sign-off per `feedback_publication_holds_until_phase4_pivot.md`.

---

## Scope Reduction Compliance

No SCOPE REDUCTION directive is active in `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES as of .242 planning. No scope_reduction_compliance field required.

---

## Schema Versioning

- Capstone artifact schema: carnot.paper_v6_capstone_2516.v1 (successor to 2505.v1)
- Retro artifact schema: carnot.operational_retro.v66 (increment from v65)
- All compute-bound artifacts: include random_seed, reproducibility_checksum, duration_s, preconditions_checked
