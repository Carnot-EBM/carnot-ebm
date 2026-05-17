# Milestone 2026.05.223 — Research Roadmap

**Pre-Test Repair, Real-Scale Live Generation, KAN-CL n=256, and KV260 RTL**

**Drafted:** 2026-05-17  
**Milestone:** 2026.05.223 (seq 223, follows 2026.05.222)  
**Experiment IDs:** exp2253–exp2265  
**Predecessor:** 2026.05.222 (FST, ODAR, CASAL)

---

## What 2026.05.222 Proved

| Component | Gate Result | Key Metric |
|-----------|-------------|------------|
| FST (Fast-Slow Training) | PASSED | sample_efficiency_ratio=4.0, kl_drift_ratio=0.0, utility_delta=0.158 |
| ActFocus + FST | PASSED | 18/20 fast-weight updates retained (0.9 retention) |
| ODAR Routing Benchmark | PASSED | 37.5% fewer tier calls, 0% accuracy loss |
| CASAL vs AdamFLIP | PASSED | mean_violation = 1.2e-17 vs 8.5e-3 (near-zero hard-constraint residuals) |
| THRML/CASAL Interface | PASSED | CASALBackend wraps SamplerBackend protocol |
| Capstone (E2E integration) | PASSED | 50% compute reduction, zero constraint residuals — BUT one-token probe only |
| KAN-CL n=256 | GATE_BLOCKED | Prior_failures rerun discipline gap |
| KV260 RTL | GATE_BLOCKED | Toolchain precondition failure (verilator/iverilog) |

**Critical gap:** The .222 capstone did a one-token GGUF probe. Full answer generation was not exercised. The FST+ODAR+CASAL stack is proven at the unit/integration level but not at the product level.

**Infrastructure gap discovered post-.222:** duplicate `test_compositional_energy` module name (one copy in `tests/python/models/`, one in `tests/python/phase3/`) causes pytest import error — the single failing test that causes `SKIP (Pre-tests failing, self-heal failed)` events in the conductor.

---

## Three Biggest Gaps vs PRD Vision

### Gap 1: Live Verification at Product Scale (FR-12)
FR-12 requires verifying full LLM outputs against constraints. The .222 capstone demonstrated interoperability with a one-token probe; a real verification pass requires generating full answers (20+ tokens) and running the complete ODAR→FST→CASAL pipeline on those answers.

**Why this is the highest priority:** Carnot's value proposition — "second-pair-of-eyes verification grounded in objective energy" — cannot be validated with a one-token probe. The entire .220-.222 arc (ActFocus, KAN-CL, AdamFLIP, FST, ODAR, CASAL) was building toward this.

### Gap 2: Continual Learning at Hardware-Portable Scale (FR-11)
KAN-CL at n=256 would demonstrate that Carnot's per-knot importance regularization scales to production-relevant constraint sizes. COOL (Springer 2026) showed 20μs update latency on embedded hardware at this scale; Carnot needs equivalent validation.

**Why this is blocked:** The .222 activation guard found that exp2247's scope could match a retired THRML parity pattern (exp2248 in the same milestone was titled "THRML/Carnot Parity at n=256" which directly matches the retired pattern "THRML/Carnot parity n=256"). This cascade-blocked exp2247 as well. The .223 plan separates these concerns: KAN-CL n=256 is purely about per-knot regularization and retention, not THRML parity.

### Gap 3: FPGA RTL Evidence (Phase 2 Hardware Path)
The KV260 hardware track requires RTL source-level evidence before any synthesis claims can be made. Verilator and Icarus Verilog must be checked as preconditions; if missing, the task should emit `blocked_toolchain_missing` rather than silently failing. Prior exp2249 was blocked due to missing toolchain check at activation time.

---

## Architecture Context

```
 ┌─────────────────────────────────────────────┐
 │           Verify-Repair Pipeline             │
 │  ┌─────────┐   ┌──────────┐   ┌──────────┐  │
 │  │  ODAR   │──▶│   FST    │──▶│  CASAL   │  │
 │  │ Routing │   │ Fast-Slow│   │ Primal-  │  │
 │  │ (EFE)   │   │ Training │   │ Dual     │  │
 │  └─────────┘   └──────────┘   └──────────┘  │
 │       │              │               │       │
 │   Fast-Path    Fast-Weight    Hard-Constraint│
 │   Skip 37-50% Context Prep   Enforcement    │
 └─────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │            KAN-CL Continual Learning         │
 │   Per-Knot Importance @ n=256 spin Ising     │
 │   Target: COOL's 20μs update latency         │
 └─────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │            Hardware Track (KV260)            │
 │   Verilator lint → Icarus sim → OSS-CAD     │
 │   synthesis (source-level claims only)       │
 └─────────────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 0: Infrastructure Repair (2 experiments)

**exp2253 — Archive .222 and Activate .223**  
Standard milestone boundary: append .222 tasks to research-complete.yaml, swap roadmap YAML, initialize changelog stub.

**exp2254 — Fix Duplicate test_compositional_energy Module**  
Root cause of the systemic `SKIP (Pre-tests failing)` events across the .216-.222 window: `tests/python/phase3/test_compositional_energy.py` and `tests/python/models/test_compositional_energy.py` share the same module name, causing pytest import collision. Fix: rename `tests/python/phase3/test_compositional_energy.py` to `tests/python/phase3/test_phase3_compositional_energy.py` and update any imports. Run full test suite to confirm single-digit warnings, zero failures.

Deliverable gate: `pretest_fixed: bool`. All Phase 1-5 tasks that were previously gate-blocked depend on this fix.

### Phase 1: Real-Scale Live Generation (3 experiments)

**exp2255 — FST+ODAR+CASAL Real-Scale Live Generation**  
Gate: `exp2254.pretest_fixed == true`.  
Run 20 full generate+verify passes on a cached SOTA GGUF model (gemma-4-26B-A4B-it-GGUF if cached, else Qwen3.6-35B-A3B-GGUF). Each pass generates a complete multi-sentence math or reasoning answer (not a one-token probe), then runs ODAR routing, FST context preparation, and CASAL constraint enforcement on the result. Gate: `n_violations_found >= 1` AND `mean_constraint_violation_after_repair < initial_mean_constraint_violation`.

This is the key validation the .222 capstone could not complete.

**exp2256 — FR-11 FST Multi-Domain Retention Validation**  
Gate: `exp2255.fst_live_validated == true`.  
FR-11 mandate (continuous self-learning). Tests FST fast-weight retention across three reasoning domains (math, code, logic) using the live-generation data from exp2255. Fast weights from Domain 1 should be partially retained when Domain 3 training begins. Gate: `cross_domain_retention_rate >= 0.75`.

**exp2257 — ODAR Real-Inference Routing Overhead Benchmark**  
Independent (no upstream gate). 100-example corpus: 50 high-confidence (low EFE, should fast-path), 50 ambiguous (high EFE, should route to deliberative). Measure: `routing_overhead_ms`, `compute_reduction_pct`, `fast_path_fraction`. Gate: `compute_reduction_pct >= 25` on a real inference workload (not the synthetic corpus of exp2244).

### Phase 2: KAN-CL n=256 (2 experiments)

**exp2258 — KAN-CL n=256 Clean Re-attempt**  
Gate: `exp2254.pretest_fixed == true`.  
Prior failure: exp2247 had `blocked_gate_check_failed` because the .222 YAML included exp2248 (THRML/Carnot Parity at n=256) which matched the retired exclusion pattern "THRML/Carnot parity n=256", cascade-blocking exp2247. This task's scope is PURELY KAN-CL per-knot importance regularization — no THRML parity measurement. The title, deliverable fields, and prompt do not reference THRML parity at all. Gate: `n256_retention_rate >= 0.85`.

**exp2259 — KAN-CL n=256 + CASAL Joint Constraint Enforcement**  
Gate: `exp2258.kancl_n256_validated == true`.  
Combine KAN-CL's per-knot importance (from exp2258) with CASAL's primal-dual sampling (from .222 exp2245). When a constraint is violated in the energy landscape, CASAL enforces hard equality while KAN-CL's importance weights guide which knots to update. Gate: `joint_constraint_satisfaction_rate >= 0.90`.

### Phase 3: KV260 RTL (2 experiments)

**exp2260 — KV260 Verilator + Icarus Simulation Clean Re-attempt**  
Gate: `exp2254.pretest_fixed == true`.  
Prior failure: exp2249 had `blocked_gate_check_failed` due to toolchain precondition not checked before conductor activation. This task adds an explicit PRECONDITIONS block checking `command -v verilator` and `command -v iverilog` BEFORE any RTL work. If tools missing: `blocked_toolchain_missing`. If tools present: run lint and simulation. Claim boundary: source-level only.

**exp2261 — OSS-CAD-Suite Synthesis from Lint-Passing RTL**  
Gate: `exp2260.lint_errors_count == 0`.  
If exp2260 achieves zero lint errors, attempt synthesis with yosys (from OSS-CAD-Suite). PRECONDITIONS check `command -v yosys`. Produces synthesis report with LUT count estimate. Claim boundary: synthesis only, no bitfile/board claim.

### Phase 4: Adversarial Null-Space Probe (1 experiment)

**exp2262 — Adversarial Null-Space Probe on k=16 + FST Stack**  
Independent (no upstream gate). Per Phase 3 defence-layer theory (Q11 TSS + CLAUDE.md Phase Prototype discipline), the integrated k=16 verifier ensemble + FST fast-weight stack must be adversarially probed. Test: construct inputs that score low-energy on ANY single verifier but high-energy on the ensemble. Gate: `ensemble_null_space_attack_rate < 0.05` (fewer than 5% of adversarial inputs fool the ensemble).

### Phase 5: Research Sweep (1 experiment)

**exp2263 — ArXiv Post-.222 Sweep + Research References Update**  
Independent (no gate). Search arxiv for papers published in May 2026 (2026-05-01 to 2026-05-17) in: energy-based verification, continual learning for KANs, Langevin constraint sampling, FPGA Ising machines, constrained LLM generation, phase 3 non-autoregressive reasoning. Add at least 3 new references to research-references.md. Produce roadmap-relevant candidate list for .224.

### Phase 6: Capstone + Retro (2 experiments)

**exp2264 — Capstone E2E Real-Scale Live Generation (.223)**  
Gates: `exp2255.fst_live_validated == true` AND `exp2258.kancl_n256_validated == true`.  
Model: opus, max_turns: 100. Full E2E run on SOTA GGUF with ODAR+FST+CASAL active: 10 math reasoning problems requiring multi-step arithmetic. KAN-CL n=256 importance weights guide which constraints to enforce. CASAL ensures hard satisfaction. FST fast weights carry cross-problem learning. Gate: `verified_repair_rate >= 0.3` (at least 3/10 problems where a violation was found and repaired).

**exp2265 — Milestone 2026.05.223 Retrospective**  
Standard retro: total_wall_time_min, n_experiments_completed, n_gate_blocks, top_successes, top_gaps, speedup_target. schema: carnot.operational_retro.v66.

---

## Dependency Graph

```
exp2253 (archive)
  │
  └──▶ exp2254 (pre-test fix)
         │
         ├──▶ exp2255 (FST live gen) ──▶ exp2256 (FR-11 multi-domain)
         │          │
         │          └──▶ exp2264 (capstone) ──▶ exp2265 (retro)
         │                   ▲
         ├──▶ exp2257 (ODAR overhead)
         │
         ├──▶ exp2258 (KAN-CL n=256) ──▶ exp2259 (KAN-CL+CASAL)
         │          │
         │          └──▶ exp2264 (capstone)
         │
         └──▶ exp2260 (KV260 RTL) ──▶ exp2261 (OSS-CAD synthesis)

exp2262 (adversarial probe) — independent
exp2263 (arxiv sweep) — independent
```

---

## Hardware Requirements

| Task | Hardware Required | Precondition Check |
|------|------------------|--------------------|
| exp2255 | GPU (RTX 3090 or gfx1150 APU) | `ls ~/.cache/huggingface/hub/` | 
| exp2256 | CPU only | None |
| exp2257 | CPU only | None |
| exp2258 | CPU only (Ising simulation) | None |
| exp2259 | CPU only | `python -c "import carnot.samplers.casal"` |
| exp2260 | None (toolchain only) | `command -v verilator` AND `command -v iverilog` |
| exp2261 | None | `command -v yosys` |
| exp2262 | CPU only | None |
| exp2264 | GPU | `ls ~/.cache/huggingface/hub/` |

---

## Exclusion Manifest Cross-Check (MANDATORY per CLAUDE.md)

Per `ops/exclusion_manifest.yaml`:

| Retired Pattern | Applies to .223? | Decision |
|-----------------|------------------|----------|
| THRML/Carnot parity n=8/16/32/64/128/256 | **NOT** in .223 | exp2258 is "KAN-CL per-knot scaling" — no THRML measurement |
| THRML/Carnot parity n=256 | **NOT** in .223 | No "THRML...parity" title in any task |
| GRPO v15 / VPRM v15 | NOT proposing | ✓ |
| WOPR puzzle cartridges | NOT proposing | ✓ |
| HardNet++/DSP repair stack | NOT proposing | ✓ |
| iCE40 PIMI | NOT proposing | ✓ |
| SpecAnn | NOT proposing | ✓ |
| exp2091 (gemini CSL grammar) | NOT proposing | ✓ |

**Result: 0 scope matches with retired experiment patterns.**

---

## Agent Routing Summary

| Task | agent_type | model | max_turns | Justification |
|------|-----------|-------|-----------|---------------|
| exp2253 | codex | gpt-5.5 | 20 | Archive pattern |
| exp2254 | codex | gpt-5.5 | 20 | Simple file rename + test run |
| exp2255 | codex | gpt-5.5 | 50 | GGUF inference + pipeline run |
| exp2256 | codex | gpt-5.5 | 30 | Training evaluation |
| exp2257 | codex | gpt-5.5 | 30 | Benchmark |
| exp2258 | codex | gpt-5.5 | 30 | Ising simulation |
| exp2259 | codex | gpt-5.5 | 30 | Integration |
| exp2260 | codex | gpt-5.5 | 30 | RTL toolchain |
| exp2261 | codex | gpt-5.5 | 30 | Synthesis |
| exp2262 | codex | gpt-5.5 | 30 | Adversarial evaluation |
| exp2263 | codex | gpt-5.5 | 20 | Literature search |
| exp2264 | (default) | opus | 100 | Multi-file GPU integration capstone |
| exp2265 | codex | gpt-5.5 | 20 | Retrospective |

All tasks default to `agent_type: codex` except exp2264 which requires multi-file tool choreography + live GPU inference → `model: opus, max_turns: 100`.

---

## Success Criteria for Milestone Completion

- [ ] Pre-test failure (duplicate test module) resolved — zero pytest errors
- [ ] Full-answer live generation executed (20 passes, >= 1 violation found + repaired)
- [ ] FR-11 multi-domain retention validated (cross_domain_retention_rate >= 0.75)
- [ ] KAN-CL n=256 retention validated (n256_retention_rate >= 0.85)
- [ ] KV260 RTL lint executed (even with warnings)
- [ ] Capstone achieves verified_repair_rate >= 0.3
- [ ] Research references updated with >= 3 new May 2026 papers

---

## .224 Setup (from gaps analysis)

If KAN-CL n=256 passes, .224 can target:
- KAN-CL n=512 scaling + hardware latency measurement
- Z1/Extropic TSU parity at n=256 (via CASALBackend interface)
- Phase 3 in-situ EBM training loop with FST online updates

If KV260 RTL passes lint:
- .224 targets synthesis + timing closure on OSS-CAD
- Vivado P&R if toolchain available

If live generation proves the stack works:
- .224 targets GSM8K benchmark (200 questions) using FST+ODAR+CASAL
- Comparison to non-FST baseline to validate the 4x sample efficiency claim at scale
