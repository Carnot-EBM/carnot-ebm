# Research Roadmap v241: Phase 4 Real-GGUF Empirical Validation + arXiv Gate + Spilled-Energy Tier 0q + PolarFire Terminal + AUROC Headline Verification

**Milestone:** 2026.05.241
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.240 — 11/12 tasks completed (group-conditional AUROC=0.975 breaches HIVE; 0.9351 TAUTOLOGY retired; Phase 4 both paths failed; PolarFire carnot installed but energy check pending; KV260 flash blocked on programmer hardware; KAN certified; FR-11 Tier 4 prototype functional; arXiv still blocked 2/4 gates)

---

## What .240 Proved

Milestone .240 had 11 of 12 tasks complete (11 artifacts confirmed in capstone, 1 exp2486 adversarially flagged for DURATION_TOO_SHORT). Key findings:

**Major wins:**
- **Group-conditional calibration AUROC=0.975** (exp2485): Separated the 9-verifier ensemble into 3 signal-type groups (logprob-class, semantic-class, logic-class), calibrated each group independently via isotonic regression, then aggregated via Fisher's method. Mean AUROC=0.975 (std=0.021, n=5 seeds) — breaches HIVE 0.9236 by +0.039 with statistical significance. This is the cite-safe paper-v6 headline replacing the retired 0.9351 (TAUTOLOGY-flagged in exp2473).
- **Simple-fusion TAUTOLOGY confirmed and 0.9351 retired** (exp2484): 5-seed replication of the original isotonic method gave true_replicated_auroc=0.7964 (95% CI [0.548, 1.045]), confirming the 0.9351 was a code artifact (same variable referenced twice). Adversarial honesty preserved.
- **KAN certified deployment** (exp2489): new_certified_coverage=0.833, new_kan_auroc=0.974, mean_local_lipschitz=2.396. KAN tier is now both the highest-AUROC verifier (0.994) AND the only certified-coverage verifier in the ensemble.
- **PolarFire carnot installed** (exp2490): carnot_runs_on_polarfire=True on riscv64 SoC after try/except JAX fix applied. **PolarFire near-terminal state** (energy_sanity_check_passed=False — one verification step remains).
- **FR-11 Tier 4 prototype functional** (exp2488): 2 knot adaptations triggered, energy_reduction_mean=2.0. All 4 FR-11 tiers have been prototyped.
- **Paper updated** (exp2492): paper_results_updated=True, phase4_section_written=True. Main.tex updated with group-conditional 0.975 AUROC + Phase 4 empirical outcomes.

**Gaps confirmed:**
1. **Phase 4 not validated by EITHER path** — ARM-EBM bijection (exp2486): pearson_r=0.108, noise floor. Qwen PRC (exp2487): used mock_model, not real GGUF inference — methodology gap, NOT a refutation. arXiv hold remains.
2. **auroc_adversarially_verified=False** — strict gate requires exp2484 hive_peer_breached (it got 0.7964). exp2485 group-conditional did breach but the strict gate is coupled to exp2484. Need to re-define the adversarial gate around exp2485 directly.
3. **PolarFire energy_sanity_check_passed=False** — carnot is installed but no Carnot energy computation has been verified on the riscv64 hardware.
4. **KV260 flash blocked** — DirtyJTAG physically incompatible (3.3V vs 1.8V), OpenOCD lacks ZynqMP PS init script. Only forward path is Digilent JTAG HS2 programmer purchase OR PYNQ SD-card boot alternative (unexplored).

---

## Three Biggest Gaps vs PRD Vision (entering .241)

### Gap 1: Phase 4 Empirical Validation (Critical Path to arXiv)

The arXiv submission operator hold requires phase4_validated_any=True. Three Phase 4 paths have been tried:
- exp2474: ODAR routing, pearson_r=0.19 — FAILED
- exp2486: ARM-EBM bijection using existing SemanticEnergy logprobs, pearson_r=0.108 — FAILED
- exp2487: Qwen PRC censorship divergence — used mock_model, NOT A REAL TEST

The PRC path (exp2487) is explicitly a methodology gap, not a refutation. The real Qwen3.6-35B-A3B-GGUF must be tested with proper PRECONDITIONS before the Phase 4 hypothesis can be declared empirically refuted.

Additionally, arXiv:2602.18671 "Spilled Energy in LLMs" provides a DIFFERENT energy metric (spilled_energy from logit distributions) that reports pearson_r 0.31-0.47 on hallucination benchmarks — substantially stronger than the ARM-EBM result. This is a genuine alternative Phase 4 path with existing telemetry data.

**Plan for .241:** Two parallel Phase 4 paths:
1. exp2496: Real Qwen3.6-35B-A3B-GGUF PRC test (precondition-gated; retire_if_same_verdict=true if real GGUF also fails)
2. exp2497: Spilled Energy CPU-only test from existing logprob fields

### Gap 2: AUROC Adversarial Verification of Group-Conditional 0.975

exp2485 group-conditional 0.975 is the cite-safe headline. But it has NOT been adversarially verified as a standalone claim (exp2484 adversarially verified the simple-fusion method, not group-conditional). The paper cannot cite 0.975 without verifying exp2485 itself is not a code artifact.

**Plan for .241:** exp2498 independently replicates exp2485 group-conditional method with explicit cross-group TAUTOLOGY check (verify Group A platt_auroc != Group B platt_auroc != Group C platt_auroc per seed).

### Gap 3: PolarFire Terminal State Completion + KV260 PYNQ Path Research

PolarFire: carnot installed, energy_sanity_check_passed=False. One SSH command needed.
KV260: bitstream generated; DirtyJTAG/OpenOCD paths ruled out; PYNQ SD-card boot is the only unexplored programmer-free alternative.

---

## Architecture Snapshot (entering .241)

```
Tier 0 Verifiers (conformal p-value ensemble):
  Tier 0a: SemanticEnergy (AUROC=0.810)
  Tier 0b: HALT (AUROC=0.8539)
  Tier 0c: FregeLogic (AUROC=0.8831)
  Tier 0d: DiffuTruth (AUROC=0.588, marginal)
  Tier 0e: LogCons Hierarchical (AUROC=0.8896)
  Tier 0f: PCIB (AUROC=0.8669)
  Tier 0g: LaaB Meta-Judgment (AUROC=0.854)
  Tier 0h: NCO (AUROC=0.678)
  Tier 0i: ODAR routing (AUROC=0.5584, excluded)
  Tier 0p: LLM-as-Judge (AUROC=0.6412, excluded)
  Group-conditional calibration (3 groups): mean AUROC=0.975 (std=0.021, n=5 seeds) [cite-safe]
  HIVE peer baseline: 0.9236

Tier 1: KAN (AUROC=0.994, certified_coverage=0.833, local_lip=2.396) [certified-deployment-ready]
Tier 2: Constraint memory (FR-11 Tier 2, SQLite, cross-session)
Tier 3: JEPA predictive verifier (FR-11 Tier 3, jepa_violation_auc=0.7633)
Tier 4: Adaptive energy landscape (FR-11 Tier 4, prototype functional, 2 adaptations)

Hardware:
  KV260: bitstream_generated (7.8MB), kv260_board_flashed=False
         DirtyJTAG incompatible (3.3V/1.8V mismatch), OpenOCD infeasible (missing PS init)
         UNEXPLORED: PYNQ SD-card boot alternative
  GateMate: TERMINAL (gatemate_bitstream_flashed=True, timing benchmark done)
  PolarFire: carnot_runs_on_polarfire=True, energy_sanity_check_passed=False (near-terminal)

Paper-v6: paper_results_updated=True (0.975 headline), audit_passed_after_fix=True
arXiv: operator hold active — 2/4 gates met; blocked on phase4_validated_any=False
Phase 1 ship: COMPLETE (PyPI+HF mirror+MCP+CLI docs+external reproducer)
```

---

## Milestone .241 Phase Structure

### Phase 0: Admin (ungated, runs immediately)
- exp2495: Archive .240 + activate .241

### Phase 1: Phase 4 Critical Path (parallel, ungated)
- exp2496: Phase 4 Qwen PRC v3 — REAL Qwen3.6-35B-A3B-GGUF (20 PRC + 20 neutral prompts; precondition-gated; retire_if_same_verdict=true if real GGUF also negative)
- exp2497: Phase 4 Spilled Energy Alternative (arXiv:2602.18671) — CPU-only from existing logprob fields; compute spilled_energy + marginalized_energy; test pearson_r vs hallucination labels on 36 telemetry corpus

### Phase 2: AUROC + New Verifiers (ungated, parallel with Phase 1)
- exp2498: AUROC Adversarial Replication v2 — independently verify exp2485 group-conditional 0.975 with cross-group TAUTOLOGY check
- exp2499: Spilled Energy Tier 0q + Conformal Ensemble v6 — implement Tier 0q from arXiv:2602.18671; integrate into group-conditional ensemble; target group_auroc > 0.975

### Phase 3: Self-Learning Completion
- exp2500: FR-11 Tier 1-4 Integration Demo — run all 4 tiers in a unified pipeline call; validate Tier 4 → Tier 1 feedback loop (continuous_self_learning_task=true)

### Phase 4: Hardware Continuity (parallel)
- exp2501: PolarFire Terminal State Closure — run IsingVerifier(n_spins=4).energy([1,-1,1,-1]) via SSH; energy_sanity_check_passed=True
- exp2502: KV260 PYNQ SD-Card Alternative Path Research — research PYNQ-Z2 SD-card boot as programmer-free bitstream loading alternative; write kv260_pynq_path.md

### Phase 5: New Verifier + Paper + Synthesis
- exp2503: Paper-v6 arXiv Final Readiness — update Phase 4 section with .241 exp2496/exp2497 results; verify AUROC claim adversarially clean; check all 4 arXiv gates
- exp2504: Curry-Howard Formal Verification Tier 0r (arXiv:2510.01069) — soft constraint verification via typed proof-path checking; new Tier 0r candidate
- exp2505: Capstone v241 — Claude+Opus synthesis: AUROC verification status, Phase 4 verdict, arXiv readiness assessment (NO HARD GATE)
- exp2506: Retro v241

---

## Dependency Graph

```
exp2495 (archive)
  |
  +-- exp2496 (Phase 4 PRC v3)  --\
  +-- exp2497 (Spilled Energy)  ---+-- exp2503 (paper readiness) --> exp2505 (capstone)
  +-- exp2498 (AUROC verify v2) --/
  +-- exp2499 (Ensemble v6)     --/
  +-- exp2500 (FR-11 integration) --> exp2505
  +-- exp2501 (PolarFire terminal) --> exp2505
  +-- exp2502 (KV260 PYNQ) --> exp2505
  +-- exp2504 (Curry-Howard) --> exp2505
                                          |
                                      exp2506 (retro)
```

exp2499 is loosely gated on exp2497 (needs spilled_energy score array to fold into ensemble). All other Phase 1-4 tasks run in parallel after exp2495.

---

## Hardware Requirements and Terminal States

| Board | Current State | Target for .241 | Terminal Condition |
|-------|--------------|-----------------|-------------------|
| KV260 | bitstream_generated, not_flashed | PYNQ SD-card path researched | kv260_board_flashed=True (requires JTAG HS2 OR PYNQ alt path confirmed) |
| GateMate | TERMINAL | optional only | n/a |
| PolarFire | carnot_installed, sanity_check_pending | energy_sanity_check_passed=True | carnot energy computation verified on riscv64 |

Hardware-Task Continuity Discipline: KV260 and PolarFire both have non-terminal tasks in .241. GateMate graduates to optional (terminal state hit in .237).

---

## FR-11 Self-Learning Mandate

exp2500 satisfies the mandatory continuous_self_learning_task per milestone. Specifically: integration of all 4 FR-11 tiers (Tier 1: online reweighting, Tier 2: SQLite memory, Tier 3: JEPA predictor, Tier 4: adaptive knot structure) in a single pipeline call. Validates the end-to-end Tier 4 → Tier 1 feedback loop.

---

## Decentralization Compliance (Rules 1-7)

- **Rule 1 (local-first):** exp2496 uses MANDATORY SOTA local GGUF (Qwen3.6-35B-A3B-GGUF). All other experiments use CPU-only computation or existing local artifacts.
- **Rule 2 (closed frontier optional):** No task requires a closed-weight model. exp2496 can fall back to gemma-4-26B-A4B-it-GGUF if Qwen not cached.
- **Rule 3 (distribution mirroring):** No new artifacts published; existing PyPI+HF+IPFS mirror structure unchanged.
- **Rule 4 (multiple surfaces):** No surface changes; existing API/CLI/MCP/HTTP surfaces maintained.
- **Rule 5 (hardware portability):** PolarFire riscv64 terminal state advances the hardware-portability claim.
- **Rule 6 (data minimization):** GGUF inference is fully local; no data leaves the machine.
- **Rule 7 (no vendor abstractions in core):** All new code (Spilled Energy, Curry-Howard, FR-11 integration) goes through abstract protocols (SamplerBackend, VerifierProtocol). No vendor imports in core.

---

## Exclusion Manifest Cross-Check

Retired patterns consulted: GRPO/VPRM, WOPR puzzles, HardNet++/DSP, THRML scaling sweep, iCE40 PIMI, SpecAnn, discriminative JEPA OOD, HalluSAE, exp2091 (CSL Grammar), exp786, exp641, exp906.

**0 retired patterns match .241 proposed tasks.** Specific checks:
- "Spilled Energy" — not retired; new paper arXiv:2602.18671 (Feb 2026)
- "Curry-Howard" — not retired; new paper arXiv:2510.01069 (ICLR 2026)
- "FR-11 integration" — not retired; exp2488 was prototype, exp2500 is integration (different deliverable)
- "PolarFire terminal" — exp2466/exp2490 succeeded (carnot installed); exp2501 is follow-on (not a retry of a failed scope)
- "Qwen PRC" — exp2487 used mock_model (methodology gap, not a retire trigger)
- "Phase 4 Spilled Energy" — no prior attempt on this specific metric/paper

---

## Agent Routing

| Task | Agent | Model | Justification |
|------|-------|-------|---------------|
| exp2495 | codex | gpt-5.5 | admin/archive pattern |
| exp2496 | codex | gpt-5.5 | GGUF inference script + precondition check; codex handles scripting |
| exp2497 | codex | gpt-5.5 | CPU-only logprob computation; mechanical numerical |
| exp2498 | codex | gpt-5.5 | 5-seed CV replication of group-conditional; mechanical |
| exp2499 | codex | gpt-5.5 | new verifier implementation + ensemble integration; mechanical |
| exp2500 | codex | gpt-5.5 | FR-11 integration pipeline; deterministic composition |
| exp2501 | codex | gpt-5.5 | SSH precondition + single energy computation |
| exp2502 | codex | gpt-5.5 | research documentation task |
| exp2503 | codex | gpt-5.5 | targeted LaTeX edits + gate check; mechanical |
| exp2504 | codex | gpt-5.5 | new verifier implementation; well-defined formal system |
| exp2505 | claude | opus | capstone synthesis: cross-artifact judgment across 10+ deliverables; requires_claude (all 3 positive criteria met) |
| exp2506 | codex | gpt-5.5 | retro template |

**Agent routing:** 11 codex/gpt-5.5 (91.7%), 1 claude+opus (8.3%). Within the CLAUDE.md Codex-Default discipline (max 2/13 claude).

---

## Failed-Experiment Rerun Compliance

| Task | Prior Exp | Prior Verdict | What Changed |
|------|-----------|---------------|--------------|
| exp2495 | exp2483 | complete: archive .239 | Different milestone metadata (archives .240 not .239) |
| exp2496 | exp2487 | complete: mock_model | Uses REAL Qwen3.6-35B-A3B-GGUF; explicit GGUF precondition check |
| exp2497 | exp2486 | complete: pearson_r=0.108 | DIFFERENT energy metric (spilled_energy, not SemanticEnergy logprob) |
| exp2498 | exp2484 | complete: 0.7964 | DIFFERENT calibration method (group-conditional, not simple isotonic) |
| exp2499 | exp2485 | complete: 0.975 | Adds NEW verifier (Tier 0q) to ensemble; 10 verifiers vs 9 |
| exp2500 | exp2488 | complete: Tier 4 prototype | Integration of ALL 4 tiers vs standalone prototype |
| exp2501 | exp2490 | complete: installed | Follow-on: energy_sanity_check (not a redo of install) |
| exp2502 | exp2491 | complete: flash_docs | DIFFERENT flash path research (PYNQ SD-card vs DirtyJTAG/OpenOCD) |
| exp2503 | exp2492 | complete: paper_updated | Final readiness check: incorporates .241 Phase 4 results + verifies AUROC |
| exp2504 | exp2451 | complete: FR-11 v5 | DIFFERENT formal method (Curry-Howard soft constraints vs NSVIF NLP) |
| exp2505 | exp2493 | complete: capstone v240 | Different milestone synthesis scope |
| exp2506 | exp2494 | complete: retro v240 | Different milestone data |
