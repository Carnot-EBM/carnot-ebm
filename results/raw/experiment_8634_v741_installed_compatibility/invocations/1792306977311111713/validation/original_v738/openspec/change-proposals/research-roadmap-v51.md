# Research Roadmap — Milestone 2026.04.51

**Title:** VR Gate Unblock + Prompt-Injection Rescue + Cascade Hardening

**CalVer:** 2026.04.51 (sequence increment from 2026.04.50)

**Authored:** 2026-04-21

**Previous Milestone:** 2026.04.50 — "Prompt-Injection Safety KAN + Structured Equation Forcing + SpecGuard Mid-Generation Verification"

---

## What Milestone 2026.04.50 Proved

Milestone .50 made structural research advances but left several critical blockers unresolved:

- **Exp 653 (StructuredEquationForcer):** SUCCEEDED — equation_forcer_ready. The COMPUTE: format forcing works; models write structured arithmetic when prompted.
- **Exp 654 (HermesV2Structured):** SUCCEEDED — hermes_v2_structured_improved. Mid-generation verification with structured equations improves detection.
- **Exp 655 (EnsembleGate v3):** gate_open=FALSE — root cause: HermesV2 recall=0.0 on mixed-format test set (structured forcing only helps on forced-equation responses, not the entire corpus); symcode=0.12 dragging ensemble to 0.224 < threshold 0.30.
- **Exp 656 (VR #18):** BLOCKED by Exp 655 gate — RETRO-033 attempt #18 never ran.
- **Exp 657 (JEPA v14 Cascade Deploy):** BLOCKED — dependency loader looked for hardcoded path that Exp 646 didn't write; Platt temperature IS available (Exp 646 = platt_calibrated) but file path mismatch.
- **Exp 658 (SpecGuard):** below_threshold — AUC < 0.70, not deployed as Tier 0f.
- **Exp 659 (FR-11 Relay):** SUCCEEDED — violations wired from Exp 656 live data.
- **Exp 660 (LSEBMCL):** SUCCEEDED — forgetting_rate=0.0 across 3 sessions, EBM replay prevents catastrophic forgetting.
- **Exp 661 (KV260 N=64):** PARTIAL — blocked_on_dfx_mgr_load_failure. Bitfile synthesized but DFX manager firmware load fails; different root cause than Vivado installation.
- **Exp 662 (Ising v3 RTL):** SUCCEEDED — ising_sampler_v3.v written with EMA inertia dynamics.
- **Exp 663 (HALP):** below_threshold — AUC < 0.75, not deployed as Tier 0g.
- **Exp 664 (DualGPU retrain):** SUCCEEDED structurally but retro_071_unresolved — GPU1 temperature rose 2C (weak signal), but DualGPU parallel forward-pass not confirmed.
- **Exp 665 (Retro):** SUCCEEDED — 6th consecutive wall-time improvement, 8.3 min/exp new project best; pre-flight test suite (519 min) now exceeds combined slowest-5 (322 min).

**Still open after .50:**
- RETRO-033: 18 consecutive VR attempts at 0% improvement (VR #18 blocked by gate)
- RETRO-070: Extraction recall ceiling — structured forcing is ready (0.20) but gate didn't open
- RETRO-071: DualGPU parallel proof still unconfirmed (14 milestones)
- RETRO-072: KV260 dfx-mgr firmware load failure (new blocker, board is powered and alive)
- RETRO-CRITICAL: Exclusion manifest wire-in — 14th consecutive milestone unconfirmed
- Prompt-injection KAN: 5th consecutive attempt (Exp 652 wrote training_curve.json but not result JSON — training ran but final write failed)

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: VR Gate Logic Blocks All Verify-Repair Progress (RETRO-033, 18 attempts)

**Root cause (confirmed from Exp 655 result):**
- `hermes_v2_recall = 0.0` on the mixed-format test set (HermesV2 only achieves recall on forced-equation responses, not prose)
- `symcode_recall = 0.12` (structural baseline, unchanged by forcing)
- `structured_recall = 0.20` (improved via COMPUTE: forcing, just below threshold)
- `ensemble_recall = 0.224 < threshold 0.30` → gate_closed

**The fix (not yet tried):** Redesign EnsembleGate v4 with structured-first logic:
- `gate_open = structured_recall >= 0.20 OR max(causal_recall, symcode_recall) >= 0.30`
- Since `causal_recall = 0.36 >= 0.30`, this gate would have opened in .50
- Exclude HermesV2 from the gate formula (it measures sentence-level recall, not equation recall)
- OR: lower gate threshold to 0.20 for structured-equation-specific experiments

This is a one-line change to the gate logic. VR #18 (RETRO-033 attempt #19) can then run.

**Experiments:** Exps 667-668 (EnsembleGate v4 redesign → Live VR #18 v2)

### Gap 2: JEPA v14 Cascade Not Deployed + Prompt-Injection KAN Still Missing

**JEPA v14 blocked (Exp 657):** Platt calibration IS done (Exp 646 = platt_calibrated), but the dependency loader used a hardcoded path that didn't match Exp 646's output. Fix: read experiment_646_*.json dynamically, extract platt_temperature, deploy to ThreeTierPipeline.

**Prompt-injection KAN (Exp 652):** Training ran (300 epochs, loss curve shows convergence), but no final result JSON was written. Root cause: the final AtomicResultWriter call was either not reached (exception in evaluation) or wrote to a path the script didn't expect. Fix: wrap evaluation in try/except with guaranteed result write.

**Experiments:** Exps 669-670 (Prompt-Injection rescue + JEPA cascade fix)

### Gap 3: Operational Overhead Exceeds Research Time (Pre-Flight Suite 519 min > Slowest-5 322 min)

**Root cause (Exp 665 retro):** The pre-flight test suite now consumes 519 estimated minutes per milestone (≈ 11.9% of total wall time), exceeding the combined slowest-5 overhead (322 min). The exclusion manifest has not been wired in for 14 consecutive milestones.

**The fix:**
1. Wire exclusion manifest BEFORE any research experiment (Exp 666 MANDATORY FIRST)
2. Implement incremental test execution: `pytest tests/python -q --co -q` to find changed-module tests, then `pytest -n auto` for parallel execution (pytest-xdist)
3. Block the chronic slowest-5 experiments from re-entering (Exp 425, 410, 308, 309, 260)

Target: pre-flight suite overhead < 100 min per milestone (80% reduction via parallelization).

**Experiment:** Exp 666 (Exclusion Manifest Wire-In v3 + Pre-Flight Compression)

---

## Architecture: Verification Cascade After Milestone 2026.04.51

```
Input: LLM response (text)
         |
[Tier 0a] CarnotThinkProbe  — generative 3-step CoT verdict
         | short-circuit if verdict='incorrect'
[Tier 0b] SpilledEnergyDetector — per-token logit-discrepancy signal
         | skip if model confident
[Tier 0c] NUPProbeV4  — contrastive energy probe (AUC=1.0 in-distribution)
         | skip if score <= nup_threshold
[Tier 0d] HallucinationBasinDetector — latent-space basin depth
         | skip if basin_risk_score <= basin_threshold
[Tier 0e] HalluField  — thermodynamic partition-function variance
         | advisory only (no early-exit)
[Tier 0f] [CANDIDATE] SpecGuard — below threshold (.50); re-evaluate with calibration
[Tier 0g] [CANDIDATE] HALP pre-generative probe — below threshold (.50); re-evaluate
[Tier 0h] LOS-Net Sequence Detector [NEW .51] — full token distribution trajectory
         | skip if sequence_risk_score <= losnet_threshold
[Tier 1 ] SinkProbe  — attention sink concentration (skip >30% at FNR 0%)
         | skip if mean_sink_score >= threshold
[Tier 2 ] EORM → JEPA v14 + Platt [DEPLOYED .51] — CoT energy reward model
         | skip if energy < eorm_threshold (Platt-calibrated thresholds)
[Tier 2.5] SymCodeVerifier  — executable Python step verification (AUC=0.804 live)
         | skip if no arithmetic violations
[Tier 2.6] HermesV2LiveLoop + StructuredEquationForcer [ACTIVE .50/.51]
         | step-by-step generation with forced equations + structured verification
[Tier 2.7] CausalReasoningVerifier  — step entailment checking (causal_recall=0.36)
         | skip if no causal violations
[Tier 3 ] Ising  — full constraint verification (0.006ms/constraint)
```

New in .51:
- Tier 2 upgrade: JEPA v14 + Platt deployed (Exp 670)
- Tier 0h candidate: LOS-Net sequence distribution detector (Exp 675)
- EnsembleGate v4: structured-first logic (Exp 667), enables VR #18 v2 (Exp 668)

---

## Dependency Graph

```
Exp 666 (manifest wire-in, MANDATORY FIRST)     — independent
Exp 667 (EnsembleGate v4 redesign)              — independent (CPU)
Exp 668 (VR #18 v2, GPU, GATED on 667)         — depends on Exp 667 gate_open=True
Exp 669 (prompt-injection rescue, GPU)          — independent
Exp 670 (JEPA v14 cascade deploy)              — depends on Exp 646 result
Exp 671 (JEPA v15 retrain, FR-11, GPU)         — depends on Exp 659 violations
Exp 672 (KV260 DFX manager fix, FPGA)          — independent (hardware)
Exp 673 (DualGPU proof v3, GPU)                — independent
Exp 674 (IAS adaptive gate calibration, CPU)   — independent
Exp 675 (LOS-Net sequence detector, CPU)        — independent
Exp 676 (MetaJuLS adaptive constraint, CPU)     — depends on Exp 668 results
Exp 677 (retrospective)                         — depends on all
```

Recommended execution order:
666 [FIRST] → 667 || 669 || 670 || 672 || 673 || 674 || 675
           → 668 (GPU, after 667 gate_open)
           → 671 (GPU, after 659 violations confirmed)
           → 676 (after 668 VR results)
           → 677 (retro)

---

## Phase Descriptions

### Phase 0: Operational Pre-Flight — Exp 666 (MANDATORY FIRST)
Wire exclusion manifest and compress pre-flight suite BEFORE any research work.
The pre-flight test suite at 519 min now exceeds the combined slowest-5 overhead (322 min).
Implement pytest-xdist parallel execution and module-level test selection.
Target: manifest wire-in confirmed (conductor_consulted=True) and suite runtime < 100 min.

### Phase 1: VR Gate Unblock — Exps 667-668
Redesign EnsembleGate to use structured-first recall logic.
Since causal_recall=0.36 already exceeds 0.30 and structured_recall=0.20 shows improvement,
the gate should open with the corrected formula.
Exp 668 (VR #18 v2) is the first RETRO-033 attempt with correct gate logic + structured forcing.
GPU REQUIRED for Exp 668.

### Phase 2: Cascade Completion — Exps 669-670
Exp 669 rescues the Prompt-Injection KAN (5th attempt) with smaller corpus and atomic writes.
Exp 670 deploys JEPA v14 + Platt calibration as default Tier 2.
Both are independent and can run in parallel.

### Phase 3: JEPA v15 Real-Data Retrain — Exp 671 (FR-11 mandatory)
Retrain JEPA on real violations from Exp 659 relay.
Use CPMI contrastive pairs + PURE min-form PRM objective (proven in Exps 577/580).
GPU REQUIRED. Target: OOD AUC >= 0.80 with ECE < 0.10.

### Phase 4: Hardware Rescue — Exps 672-673
Exp 672: KV260 DFX manager protocol fix — diagnose and resolve dfx-mgr-client load failure.
The bitfile (N=64) is synthesized; the board is powered. A firmware loading protocol issue
(not Vivado) is blocking the final hardware test.
Exp 673: DualGPU proof v3 — 14th milestone without confirmed GPU1 compute. Pre-download
Qwen2.5-7B-Instruct, run ThreadPoolExecutor parallel inference, measure GPU1 utilization.

### Phase 5: New Research — Exps 674-676
Exp 674: IAS adaptive gate calibration (arXiv 2506.09338). Quantile regression on FOVER pairs
sets gate thresholds adaptively, addressing the structural reason fixed thresholds keep failing.
Exp 675: LOS-Net sequence distribution detector (arXiv 2503.14043). Lightweight attention model
over full token distribution trajectory. If AUC >= 0.75, proposes Tier 0h.
Exp 676: MetaJuLS adaptive constraint propagation (arXiv 2601.00095). Online meta-RL adapter
for StructuredEquationForcer; updates extraction policy from live feedback.

### Phase 6: Retrospective — Exp 677
Standard milestone retrospective following the carnot.operational_retro schema.

---

## Open RETROs Being Addressed

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-033 (18 consecutive VR 0%) | CRITICAL | Exp 667 (gate redesign) + Exp 668 (VR #18 v2) |
| RETRO-070 (extraction ceiling 12%) | CRITICAL | Exp 667 (structured-first gate logic) |
| RETRO-071 (DualGPU unconfirmed) | HIGH | Exp 673 (DualGPU proof v3) |
| RETRO-072 (KV260 dfx-mgr failure) | HIGH | Exp 672 (DFX protocol fix) |
| RETRO-CRITICAL (manifest wire-in) | CRITICAL | Exp 666 (MANDATORY FIRST) |
| Prompt-injection KAN (5th fail) | MEDIUM | Exp 669 (rescue v2, reduced corpus) |
| JEPA cascade blocked (path mismatch) | MEDIUM | Exp 670 (dynamic path loading) |

---

## Success Criteria

| Criterion | Target | Gated On |
|-----------|--------|---------|
| manifest_consulted=True | conductor reads exclusion JSON | Exp 666 |
| preflight_suite_runtime < 100 min | pytest-xdist parallel | Exp 666 |
| gate_open=True | structured_recall >= 0.20 OR causal_recall >= 0.30 | Exp 667 |
| vr_18_v2_signed_improvement > 0 | first positive VR on live GPU | Exp 668 |
| prompt_injection_auroc >= 0.90 | KAN distilled from gpt-oss-safeguard | Exp 669 |
| jepa_v14_deployed=True | Platt temperature wired into ThreeTierPipeline | Exp 670 |
| jepa_v15_ood_auc >= 0.80 | real violations + CPMI+PURE | Exp 671 |
| kv260_hardware_latency_us < 100 | dfx-mgr resolved + bitfile loaded | Exp 672 |
| gpu1_utilization > 0 | ThreadPoolExecutor parallel inference | Exp 673 |
| ias_gate_open_on_550 = True | adaptive threshold vs .50 test set | Exp 674 |
| losnet_auc >= 0.75 | sequence distribution detector | Exp 675 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 666 | CPU only | Manifest wire-in, test compression |
| Exp 667 | CPU only | Gate logic redesign, reads result files |
| Exp 668 | 2x RTX 3090 | CARNOT_FORCE_LIVE=1, GPU REQUIRED |
| Exp 669 | 1x RTX 3090 | Teacher model already cached (models/gpt-oss-safeguard-20b/) |
| Exp 670 | CPU only | Reads Exp 646 result, writes ThreeTierPipeline config |
| Exp 671 | 1x RTX 3090 | GPU REQUIRED, CARNOT_FORCE_LIVE=1 |
| Exp 672 | KV260 FPGA | Board powered, bitfile synthesized; dfx-mgr fix needed |
| Exp 673 | 2x RTX 3090 | Proof requires both GPUs active simultaneously |
| Exp 674 | CPU only | Quantile regression on FOVER pairs |
| Exp 675 | CPU only | Operates on pre-computed logit distributions |
| Exp 676 | CPU only | Meta-RL policy updates from live logs |
| Exp 677 | CPU only | Retrospective analysis |

---

## New Papers Incorporated (arXiv Scan 2026-04-21)

| Paper | Title | Filed As |
|-------|-------|---------|
| arXiv 2506.09338 | IAS: Instance-Adaptive Scaling for PRM Calibration | Exp 674 |
| arXiv 2503.14043 | LOS-Net: Sequence-Level Hallucination Detection | Exp 675 |
