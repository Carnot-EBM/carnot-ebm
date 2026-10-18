# Research Roadmap — Milestone 2026.05.225

**Title:** PyPI Escalation Pre-Test Fix, FST Live Generation, KAN-CL n=256, Eidoku CSP Gate, and Projected-Langevin Baseline

**Milestone:** 2026.05.225
**Date:** 2026-05-17
**Experiment IDs:** exp2280–exp2293 (14 tasks)
**Previous milestone:** 2026.05.224

---

## What Milestone 2026.05.224 Proved

Milestone .224 completed 2 of 14 tasks and diagnosed a NEW cascade root cause that blocked the remaining 12:

1. **Archive step failed** (exp2266): `honest_verdict: "complete: blocked_roadmap_missing"` — the archive task checked for `milestone: "2026.05.223"` in `research-roadmap.yaml`, but the file already shows `"2026.05.224"` (the planning agent wrote .224 tasks directly to the active roadmap). Fix for .225: check for `"2026.05.224"` in the precondition.

2. **DualGPU fix partially succeeded but revealed a NEW blocker** (exp2267): `python/carnot/inference/__init__.py` now exports the four DualGPU symbols correctly. However, the full pre-test suite fails on `tests/python/test_pypi_escalation.py` which imports `check_pypi_escalation` and `run_escalation` from `carnot.pypi_escalation` — but `python/carnot/pypi_escalation.py` only defines `generate_escalation_report` and `main`. These two missing functions are the NEW cascade root cause.

3. **All remaining tasks blocked** (exp2268–exp2279): 12 tasks cascaded to `blocked_gate_check_failed` or were never activated because the pre-test suite fails whenever `tests/python/test_pypi_escalation.py` is collected. None of the Phase 1/2/3 experiments ran. Experiments exp2269, exp2271, exp2273, exp2275, exp2276, exp2277, exp2278, and exp2279 have no artifact files at all.

**Key gaps entering .225:**
- `carnot.pypi_escalation` missing `check_pypi_escalation` and `run_escalation` — the pre-test cascade root cause.
- Archive task precondition mismatch — fixed in exp2280 by checking for `"2026.05.224"`.
- FST+ODAR+CASAL live generation has never achieved full-answer (>50 tokens) evidence (blocked 3+ milestones).
- KAN-CL n=256 validation blocked for 4 consecutive milestones (.221, .222, .223, .224).
- Eidoku CSP gate (exp2275), Projected-Langevin baseline (exp2276), and ArXiv sweep (exp2277) never ran.
- Adversarial null-space probe (exp2274) blocked by cascade despite having no structured gate.

---

## Architecture Snapshot

```
Milestone 2026.05.225 additions (bold = new this milestone):

Infrastructure:
  **carnot.pypi_escalation: add check_pypi_escalation + run_escalation** (exp2281)
  → fixes the second pre-test cascade; unblocks ALL subsequent tasks

Verify-Repair Cascade:
  Tier 0a–0e probes → ODAR routing → Tier 1–2.7 → Tier 3 Ising
                                            ^
                             **Eidoku CSP gate** (exp2289)
                             Tier 2.8: context-calibrated cost threshold
                             rejects smooth falsehoods via structural CSP
                             (first run — was blocked in .224)

Constrained Sampling:
  **Projected-Langevin sampler** (exp2290): arXiv:2605.05387
  ProjectedLangevinSampler in python/carnot/samplers/projected_langevin.py
  Compared against CASAL on 3 synthetic constrained problems
  (first run — was blocked in .224)

Continuous Self-Learning (FR-11):
  FST slow/fast weight decomposition → **multi-domain retention** (exp2283)
  Fast weights persist across 3 reasoning domains (math→code→logic)
  cross_domain_retention_rate >= 0.75 closes FR-11 multi-domain claim

Hard Constraint Sampling:
  CASAL + KAN-CL per-knot importance jointly (exp2285)
  **joint_constraint_satisfaction_rate >= 0.90** closes integration loop

KAN Tier:
  KAN-CL per-knot importance → **n=256 validation** (exp2284)
  n256_retention_rate >= 0.85 closes the 4-milestone carry-forward

Hardware Track (active):
  KV260 RTL → **Verilator lint + Icarus sim** (exp2286, retry)
  OSS-CAD → **Yosys synthesis** (exp2287, gated on exp2286.lint_errors_count==0)

Defence Layer (Phase 3):
  k=16 ensemble → **adversarial null-space probe** (exp2288, retry)
  ensemble_fooled_rate < 0.05 validates AND-composition security property
```

---

## The 3 Biggest Gaps vs PRD Vision

1. **Pre-test cascade blocks all meaningful experiments (4th consecutive milestone)** — The root cause shifted from DualGPU ImportError (.224) to `carnot.pypi_escalation` missing symbols. `tests/python/test_pypi_escalation.py` imports `check_pypi_escalation` and `run_escalation`, but the module only defines `generate_escalation_report` and `main`. Adding these two functions unblocks the full pre-test suite and cascades to all subsequent tasks.

2. **FST live answer generation never validated (4+ milestones)** — The Phase 1 mandate (FR-11 continuous self-learning via Fast-Slow Training) requires full-answer generation evidence (>50 tokens). Single-token probes have been accepted as surrogates but fail the anti-fabrication gate. This is gated on the pre-test fix (exp2281) and runs immediately once the cascade is broken.

3. **KAN-CL n=256 validation blocked 4 consecutive milestones** — Per-knot importance regularization at n=256 is the load-bearing gate for the COOL-parity claim and the hardware story (COOL: 20μs update latency validated in Springer 2026). Every prior milestone had an exogenous blocker. With the pre-test fixed, this is mechanically straightforward.

---

## Phase Structure

### Phase 0: Infrastructure (mandatory, no gate dependencies)

| Task | Scope | Gate |
|------|-------|------|
| exp2280 | Archive .224, activate .225 | `archive_ready: true` |
| exp2281 | Fix `carnot.pypi_escalation` missing `check_pypi_escalation` + `run_escalation` | `pretest_fixed: true` |

### Phase 1: Carry-forward FST + KAN-CL (gated on exp2281.pretest_fixed)

| Task | Scope | Gate |
|------|-------|------|
| exp2282 | FST+ODAR+CASAL Real-Scale Live Generation (retry exp2268) | `fst_live_validated: true` |
| exp2283 | FR-11 FST Multi-Domain Retention (FR-11 mandate, retry exp2269) | `fr11_multidomain_passed: true` |
| exp2284 | KAN-CL n=256 Per-Knot Clean Re-attempt (retry exp2270) | `kancl_n256_validated: true` |
| exp2285 | KAN-CL n=256 + CASAL Joint (retry exp2271) | `kancl_casal_joint_validated: true` |

### Phase 2: Hardware (gated on exp2281.pretest_fixed)

| Task | Scope | Gate |
|------|-------|------|
| exp2286 | KV260 RTL Verilator Lint + Icarus Sim (retry exp2272) | `lint_errors_count == 0` |
| exp2287 | OSS-CAD Yosys Synthesis (retry exp2273) | synthesis report generated |

### Phase 3: New Science + Adversarial (gated on exp2281.pretest_fixed or ungated)

| Task | Scope | Gate |
|------|-------|------|
| exp2288 | Adversarial Null-Space Probe k=16 (retry exp2274) | `adversarial_probe_passed: true` |
| exp2289 | Eidoku CSP Gate (arXiv:2512.20664, new — was exp2275) | `eidoku_gate_validated: true` |
| exp2290 | Projected-Langevin Baseline (arXiv:2605.05387, new — was exp2276) | `projected_langevin_competitive: true` |

### Phase 4: Research + Capstone + Retrospective

| Task | Scope | Gate |
|------|-------|------|
| exp2291 | ArXiv Post-.224 Research Sweep | `n_new_papers_found >= 3` |
| exp2292 | Capstone E2E (gated on exp2282 + exp2284) | `capstone_passed: true` |
| exp2293 | Milestone .225 Retrospective | `honest_verdict: complete:` |

---

## Dependency Graph

```
exp2280 (archive) ─────────────────────────────────────────────────────────
                                                                            │
exp2281 (pypi-fix) ──────────────────────────────────────────────────────┐ │
    │                                                                     │ │
    ├─→ exp2282 (fst-live-gen) ──→ exp2283 (fr11-multidomain) [FR-11]    │ │
    │                                                                     │ │
    ├─→ exp2284 (kancl-n256) ──→ exp2285 (kancl-casal-joint)             │ │
    │                                                                     │ │
    ├─→ exp2286 (kv260-rtl) ──→ exp2287 (yosys)                          │ │
    │                                                                     │ │
    └─→ exp2288 (adversarial-probe)                                       │ │
                                                                          │ │
exp2289 (eidoku-csp) [ungated]                                            │ │
exp2290 (projected-langevin) [ungated]                                    │ │
exp2291 (arxiv-sweep) [ungated]                                           │ │
                                                                          │ │
exp2292 (capstone) ← exp2282.fst_live_validated + exp2284.kancl_n256_val ─┘ │
exp2293 (retro) ────────────────────────────────────────────────────────────┘
```

---

## Continuous Self-Learning Mandate (FR-11)

Satisfied by **exp2283** (FR-11 FST Multi-Domain Retention):
- `continuous_self_learning_task: true` in artifact contract
- Gate: `cross_domain_retention_rate >= 0.75`
- Validates FST fast weights generalize across domains (math→code→logic)
- Gated on exp2282 (live generation evidence must exist before multi-domain eval)

---

## Hardware Requirements

- **exp2282, exp2283, exp2292**: Cached SOTA GGUF (one of: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`) — blocked if not cached
- **exp2284, exp2285, exp2288**: CPU-only (JAX, no GPU required)
- **exp2286, exp2287**: Verilator + iverilog + yosys (blocked if absent, honest verdict)
- **exp2289, exp2290**: CPU-only (numpy, no GPU required)

---

## Decentralization Check (CLAUDE.md Rules 1–7)

- Rules 1–2: All GGUF tasks use open-weight local models; no closed-weight dependency
- Rule 3: No new artifacts published this milestone (retried experiments only)
- Rule 4: No integration surface drift (all four surfaces unchanged)
- Rule 5: KV260 hardware track continues; FPGA path intact
- Rules 6–7: No vendor SDK imports added to core

All seven rules satisfied.

---

## Cross-References

- `ops/exclusion_manifest.yaml` — confirmed: none of the 14 tasks match retired experiment scope
- `research-references.md` — Eidoku (arXiv:2512.20664), Projected-Langevin (arXiv:2605.05387), KAN-CL (arXiv:2605.12306), CASAL (arXiv:2505.18017), FST (arXiv:2605.12484)
- `ops/known-issues.md` — no active SCOPE REDUCTION directive; no mandatory-next-milestone priorities pending 3+ milestones
- `_bmad/architecture.md` — Last Reconciled 2026-05-16 (< 30 days, no flag needed)
