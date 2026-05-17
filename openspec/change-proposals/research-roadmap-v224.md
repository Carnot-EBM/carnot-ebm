# Research Roadmap — Milestone 2026.05.224

**Title:** DualGPU ImportError Fix, FST Live Generation, KAN-CL n=256 Validation, and Eidoku CSP Gate

**Milestone:** 2026.05.224
**Date:** 2026-05-17
**Experiment IDs:** exp2266–exp2279 (14 tasks)
**Previous milestone:** 2026.05.223

---

## What Milestone 2026.05.223 Proved

Milestone .223 completed 5 of 13 tasks and identified one cascade root cause that blocked the remaining 8:

1. **ODAR real-inference benchmark validated** (exp2257): `compute_reduction_pct=37.5`, `routing_overhead_ms=0.015ms` on 100 real EFE-routed examples. Compute reduction is real and overhead is negligible.
2. **Post-.222 arXiv sweep completed** (exp2263): 8 new papers found, including projected-Langevin equality constraints (arXiv:2605.05387), p-bit FPGA annealer (arXiv:2602.16143), and geometric hallucination taxonomy (arXiv:2602.13224).
3. **Retrospective completed** (exp2265): root cause fully diagnosed.

**Root cause of cascade:** `tests/python/test_dual_gpu.py` imports `DualGPUExecutionResult` from `carnot.inference`, but `python/carnot/inference/__init__.py` is empty — the class exists in `dual_gpu.py` but is never re-exported. The conductor's pre-test check hits this ImportError and labels ALL tasks (even those with no structured gate) as `blocked_gate_check_failed` or SKIP. This one-line fix (add the export to `__init__.py`) unblocks 10 of the 13 .223 tasks.

**Key gaps entering .224:**
- DualGPU ImportError is the cascade root cause — highest priority fix.
- KAN-CL n=256 validation blocked for 3 consecutive milestones (.221, .222, .223).
- Full live answer generation (>50 tokens, not single-token probe) never achieved.
- Adversarial null-space probe (exp2262) blocked by cascade.
- KV260 RTL Verilator lint blocked by cascade.

---

## Architecture Snapshot

```
Milestone 2026.05.224 additions (bold = new):

Infrastructure:
  **carnot.inference.__init__ re-exports DualGPU symbols** (exp2267)
  → unblocks all pre-test-gated tasks

Verify-Repair Cascade:
  Tier 0a–0e probes → ODAR routing → Tier 1–2.7 → Tier 3 Ising
                                            ^
                             **Eidoku CSP gate** (exp2275)
                             Tier 2.8: context-calibrated cost threshold
                             rejects smooth falsehoods via structural CSP

Continuous Self-Learning (FR-11):
  FST slow/fast weight decomposition → **multi-domain retention** (exp2269)
  Fast weights persist across 3 reasoning domains (math→code→logic)
  cross_domain_retention_rate >= 0.75 gates FR-11 multi-domain claim

Hard Constraint Sampling:
  CASAL primal-dual + KAN-CL per-knot importance (jointly, exp2271)
  **joint_constraint_satisfaction_rate >= 0.90** closes integration test

KAN Tier:
  KAN-CL per-knot importance → **n=256 validation** (exp2270)
  n256_retention_rate >= 0.85 closes the 3-milestone carry-forward

Hardware Track (active):
  KV260 RTL → **Verilator lint + Icarus sim** (exp2272, retry)
  OSS-CAD → **Yosys synthesis from lint-clean RTL** (exp2273)

Defence Layer (Phase 3):
  k=16 ensemble → **adversarial null-space probe** (exp2274)
  ensemble_fooled_rate < 0.05 validates AND-composition security property

New Science (.224):
  **Eidoku CSP gate** (exp2275): arXiv:2512.20664 neuro-symbolic System-2
  **Projected-Langevin equality baseline** (exp2276): arXiv:2605.05387
```

---

## The 3 Biggest Gaps vs PRD Vision

1. **Cascade root cause (DualGPU ImportError) blocks all meaningful experiments** — `carnot.inference.__init__` is empty; the `DualGPUExecutionResult` class lives in `dual_gpu.py` but is never exported. The conductor's pre-test check imports from `carnot.inference`, triggers an ImportError, and labels all tasks as blocked. Fix: add four lines to `__init__.py`. This is the highest-leverage action in .224.

2. **KAN-CL n=256 never validated (3 consecutive milestones)** — Per-knot importance regularization at n=256 scale is the load-bearing gate for the COOL-parity claim and the KAN hardware story. It's been blocked by cascade (.221 pre-test churn, .222 THRML contamination, .223 DualGPU ImportError). With exp2267 fixing the ImportError, exp2270 should finally run.

3. **Live full-answer generation never proven** — All capstones since .221 have either been blocked or used single-token probes as "live evidence." The FST+ODAR+CASAL integrated stack exists but has never been validated end-to-end with full multi-sentence answers. Closing this gap is the primary product-level credibility claim.

---

## Phase Descriptions

### Phase 0: Archive and DualGPU Fix
Two tasks. Archive .223 (exp2266) and fix the DualGPU ImportError cascade root cause (exp2267). The fix adds `DualGPUExecutionResult`, `DualGPURunner`, `estimate_model_size_billions`, and `requires_device_map_auto` to `python/carnot/inference/__init__.py`. Sets `pretest_fixed: true` when pytest runs clean. max_turns=20 each.

### Phase 1: FST Live Generation + FR-11 + KAN-CL (4 tasks)
All four gated on `exp2267.pretest_fixed == true`. Two parallel lanes:

**Lane A (FST live generation):**
- exp2268: Run 20 full-answer generation passes (>50 tokens each) on cached SOTA GGUF with ODAR+FST+CASAL. Sets `fst_live_validated: true` when violations found in real answers.
- exp2269: FR-11 multi-domain retention across math→code→logic domains. Sets `fr11_multidomain_passed: true` when `cross_domain_retention_rate >= 0.75`. Contains `continuous_self_learning_task: true`.

**Lane B (KAN-CL):**
- exp2270: KAN-CL n=256 per-knot importance. Sets `kancl_n256_validated: true` when `n256_retention_rate >= 0.85`.
- exp2271: KAN-CL + CASAL joint constraint enforcement. Sets `kancl_casal_joint_validated: true` when `joint_constraint_satisfaction_rate >= 0.90`.

### Phase 2: Hardware Track (2 tasks)
Both gated on `exp2267.pretest_fixed == true`. Retry of .223 hardware tasks:
- exp2272: KV260 RTL Verilator lint + Icarus 10-cycle simulation. Explicit toolchain PRECONDITIONS (check verilator AND iverilog before RTL work). Sets `lint_errors_count` and `simulation_cycles_completed`.
- exp2273: OSS-CAD Yosys synthesis from lint-clean RTL. Gated on `exp2272.lint_errors_count == 0`. Records `lut_count_estimate`.

### Phase 3: Defence + New Science (3 tasks)
- exp2274: Adversarial null-space probe on k=16 ensemble + FST stack. No gate (retry of .223 exp2262 which was cascade-blocked). 200 adversarial inputs, validates `ensemble_fooled_rate < 0.05`.
- exp2275: Eidoku CSP verification gate implementation (arXiv:2512.20664). Implement context-calibrated cost threshold as Tier 2.8 verifier in the cascade. Evaluate on 50 synthetic reasoning examples.
- exp2276: Projected-Langevin equality constraint baseline (arXiv:2605.05387). Implement minimal projected-Langevin sampler for linear equality constraints and compare against CASAL on synthetic constrained tasks.

### Phase 4: Research Sweep, Capstone, and Retro (3 tasks)
- exp2277: Post-.223 arXiv sweep + references update. Searches 6 query strings, adds >= 3 new papers.
- exp2278: Capstone E2E (model=opus, max_turns=100). Gated on BOTH `exp2268.fst_live_validated == true` AND `exp2270.kancl_n256_validated == true`. Full 10-problem verify-repair pass with SOTA GGUF, mean_answer_length_tokens >= 50, verified_repair_rate >= 0.3.
- exp2279: Retrospective (max_turns=20).

---

## Dependency Graph

```
exp2266 (archive .223)
    |
exp2267 (DualGPU fix → pretest_fixed)
    |
    +---> exp2268 (FST live gen → fst_live_validated)
    |         |
    |         +---> exp2269 (FR-11 multidomain)
    |
    +---> exp2270 (KAN-CL n=256 → kancl_n256_validated)
    |         |
    |         +---> exp2271 (KAN-CL+CASAL joint)
    |
    +---> exp2272 (KV260 RTL → lint_errors_count)
              |
              +---> exp2273 (synthesis, gated lint==0)

exp2274 (adversarial probe, no gate)
exp2275 (Eidoku CSP gate, no gate)
exp2276 (projected-Langevin, no gate)
exp2277 (arXiv sweep, no gate)

exp2268 (fst_live_validated) +---+
                                  |---> exp2278 (capstone, opus)
exp2270 (kancl_n256_validated) +--+

exp2279 (retro, no gate)
```

---

## Hardware Requirements

| Task | Hardware | Resource |
|------|----------|----------|
| exp2268, exp2278 | GPU (RTX 3090) | GGUF inference (>50 tokens) |
| exp2272, exp2273 | CPU only | Verilator/Icarus/Yosys |
| exp2270, exp2271 | CPU only | JAX synthetic spin models |
| exp2274–exp2276 | CPU only | Synthetic benchmark |
| all others | CPU only | — |

---

## Carry-Forward Discipline

Tasks with `.223 carry-forwards must cite their prior failure explicitly:
- exp2268 cites exp2255 (pretest_fixed=false, root cause = DualGPU ImportError)
- exp2269 cites exp2256 (blocked because upstream exp2255 gate-blocked)
- exp2270 cites exp2258 (pretest_fixed=false, same root cause)
- exp2271 cites exp2259 (blocked because upstream exp2258 gate-blocked)
- exp2272 cites exp2260 (pretest_fixed=false, same root cause)
- exp2273 cites exp2261 (blocked because upstream lint not clean)
- exp2274 cites exp2262 (blocked_gate_check_failed, pre-test cascade blocked activation)
- exp2278 cites exp2264 (both upstream gates false due to .223 cascade)

---

## Exclusion Manifest Cross-Check

Verified against `ops/exclusion_manifest.yaml` before proposing tasks. None of the 14 tasks match retired patterns:
- No THRML scaling sweep (blocked_patterns: "THRML/Carnot parity n=*")
- No GRPO/VPRM v15
- No WOPR puzzle cartridges
- No HardNet++/DSP
- No SpecAnn/Spectral Annealing
- No iCE40 PIMI

---

## Scope-Reduction Compliance

No SCOPE REDUCTION directive is active in `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES as of 2026-05-17. Normal research-breadth milestone.

---

## New Literature Incorporated

| Paper | arXiv | .224 Experiment |
|-------|-------|-----------------|
| Eidoku: Neuro-Symbolic Verification Gate | 2512.20664 | exp2275 |
| Conditional Diffusion Under Linear Constraints | 2605.05387 | exp2276 |
| Generative Thermodynamic Computing | 2506.15121 | future (.225+) |
| Hard Constraints Meet Soft Generation | 2602.01090 | future (.225+) |
| Constrained Language Generation with Discrete Diffusion | 2503.09790 | future (.225+) |
