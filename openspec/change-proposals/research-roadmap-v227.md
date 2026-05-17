# Research Roadmap — Milestone 2026.05.227

**Title:** Pre-Test Fix Completion, FST Live Generation v7, NSVIF First Run, ML-Assisted Ising Machines

**Milestone:** 2026.05.227
**Date:** 2026-05-17
**Experiment IDs:** exp2308–exp2321 (14 tasks)
**Previous milestone:** 2026.05.226

---

## What Milestone 2026.05.226 Proved

Milestone .226 was a partial breakthrough after six consecutive blocked milestones. The Claude Sonnet + C+E Opus escalation approach (exp2295, `requires_claude: true`) succeeded where five codex attempts had failed:

1. **pypi_escalation ImportError is RESOLVED** (exp2295): `check_pypi_escalation` and `run_escalation` symbols were added to `python/carnot/pypi_escalation.py`. All 4 pypi_escalation-specific tests now pass (errors: 0, failures: 0).

2. **Two pre-existing test failures remain** (exp2295): The full suite reports `full_pretest_failures: 2` — two tests that existed before the ImportError cascade and were masked by it:
   - `tests/python/test_experiment_1347_thrml_compatibility_parity_audit.py::test_req_sample_041_probe_reports_direct_import_success_without_version`
   - `tests/python/test_experiment_1182_paper_v5_medium_low_issues_11_18.py::TestIssue11ThroughIssue15::test_issue_14_soskan_aurocs_have_corpus_and_n`
   
   These are pre-existing (not introduced by .226) and unrelated to pypi_escalation. `pretest_fixed: false` because `full_pretest_failures != 0`.

3. **All gated tasks still gate-blocked** (exp2296-exp2307): Because `pretest_fixed: false`, all structured gates failed. The conductor's system-wide pre-test check also blocked the ungated tasks before they could run.

**Key change entering .227:**
- The pypi_escalation cascade is broken. The remaining blocker is 2 specific test failures with known names.
- exp2309 targets EXACTLY those 2 failures — a well-defined debugging task with known scope.
- Once exp2309 achieves `pretest_fixed: true`, all Phase 1–3 work that has been queued for 6+ milestones can finally execute.

---

## Architecture Snapshot

```
Milestone 2026.05.227 additions (bold = new this milestone):

Infrastructure:
  carnot.pypi_escalation: FIXED in .226 (exp2295)
  **2 remaining pre-test failures FIXED** (exp2309)
  → test_req_sample_041_probe_reports_direct_import_success_without_version
  → test_issue_14_soskan_aurocs_have_corpus_and_n
  → Once exp2309.pretest_fixed=true, 12 queued experiments can execute

Verify-Repair Cascade:
  Tier 0a–0e probes → ODAR routing → Tier 1–2.7 → Tier 2.8 Eidoku CSP (exp2315) → Tier 3 Ising
  (Eidoku queued since .224; first actual run in .227 after pre-test fix)

  NSVIF Z3 extractor (exp2313) — FIRST ACTUAL RUN
    arXiv:2601.17789 — PRD Priority #1: rebuild constraint extraction
    LogicalStepExtractor (python/carnot/pipeline/extract.py)
    logical-step parser → Z3 prover → nsvif_tpr >= 0.50 gate
    Queued since .225; now unblocked

  VERGE MCS repair (exp2314) — FIRST ACTUAL RUN
    arXiv:2601.20055 — SMT Minimal Correction Subsets
    VERGERepair class (python/carnot/verify/verge_repair.py)
    Queued since .225; now unblocked

Constrained Sampling:
  CASAL (primal-dual) — already implemented (.222)
  Projected-Langevin sampler (exp2316) — FIRST ACTUAL RUN
    arXiv:2605.05387, comparison vs CASAL on 3 problems
    Queued since .224; now unblocked

Continuous Self-Learning (FR-11):
  FST slow/fast weight decomposition → multi-domain retention v4 (exp2311)
  Gated on exp2310.fst_live_validated (live generation must work first)
  cross_domain_retention_rate >= 0.75 closes FR-11 multi-domain claim

KAN Tier:
  KAN-CL per-knot importance → n=256 validation v6 (exp2312)
  n256_retention_rate >= 0.85 closes the 6-milestone carry-forward

Hardware Track (active):
  KV260 RTL → Verilator lint + Icarus sim v6 (exp2317, retry)
  **ML-Assisted Ising Machines** (exp2318) — NEW from arXiv:2503.23966
    ML heuristic initialization for parallel Ising sampler
    Measures convergence speedup on 3 problem sizes (n=64/128/256)
    Candidate complement to KV260 RTL: better initial states → fewer sweeps

Defence Layer (Phase 3):
  k=16 ensemble → adversarial null-space probe v5 (exp2319, retry)
  ensemble_fooled_rate < 0.05 validates AND-composition security
```

---

## The 3 Biggest Gaps vs PRD Vision

1. **Pre-test cascade still blocks substantive work (6th consecutive milestone)** — exp2295 made decisive progress: the pypi_escalation ImportError is fixed, 2 specific pre-existing failures remain known. exp2309 is the targeted fix for those exact 2 failures. The task scope is radically narrower than any prior pre-test fix task: 2 named tests, likely simple assertion failures in LaTeX content or module wiring. With `requires_claude: true, max_turns: 40`, this should succeed.

2. **FR-11 continuous self-learning has no live-model evidence** — Six milestones of cascades have left FR-11 with only synthetic training evidence. FST multi-domain retention (exp2311) is gated on live generation (exp2310). Once the pre-test is fixed, this critical path (exp2309 → exp2310 → exp2311) finally runs end-to-end.

3. **PRD Priority #1 (constraint extraction) has no implementation** — Research-program.md lists "Rebuild constraint extraction for real models" as the HIGHEST PRIORITY since 2026-04-11. NSVIF (exp2313) is the first direct implementation. It has never run due to cascades. This milestone finally delivers it.

---

## Phase Structure

### Phase 0: Archive + Pre-Test Fix Completion (exp2308–exp2309)

**exp2308** — Archive .226 and activate .227 (codex, max_turns: 20)
- Precondition: research-roadmap.yaml contains "2026.05.226"
- Archives .226 to research-complete.yaml
- Replaces research-roadmap.yaml with research-roadmap-next.yaml

**exp2309** — Fix 2 Remaining Pre-Test Failures (requires_claude: true, max_turns: 40)
- Targets EXACTLY 2 named failing tests from exp2295 artifact
- Diagnoses root cause (assertion mismatch, missing field, stale content)
- Fixes the code or test content
- Confirms full pytest suite: errors=0, failures=0
- Sets pretest_fixed: true (gates all Phase 1–3 tasks)

### Phase 1: Live Generation + Continuous Self-Learning (exp2310–exp2312)

**exp2310** — FST+ODAR+CASAL Live Generation v7 (codex, gated on exp2309.pretest_fixed)
- 20 full-answer generations (>50 tokens each) on SOTA GGUF
- PRECONDITIONS: GGUF cache check, module imports, no fabrication
- Sets fst_live_validated: true (gates exp2311 + exp2320)

**exp2311** — FR-11 FST Multi-Domain Retention v4 (codex, gated on exp2309.pretest_fixed AND exp2310.fst_live_validated)
- 3-domain sequential FST training: math → code → logic
- cross_domain_retention_rate >= 0.75 gate
- continuous_self_learning_task: true (FR-11 mandate)

**exp2312** — KAN-CL n=256 Per-Knot Retention v6 (codex, gated on exp2309.pretest_fixed)
- n=256 sequential Ising domain training
- n256_retention_rate >= 0.85 gate (closes 6-milestone carry-forward)
- kancl_n256_validated: true (gates exp2320)

### Phase 2: Neuro-Symbolic Extraction + New Techniques (exp2313–exp2316)

**exp2313** — NSVIF LogicalStepExtractor v2 (codex, max_turns: 40, gated on exp2309.pretest_fixed)
- PRD Priority #1: first actual implementation of Z3-based constraint extraction
- LogicalStepExtractor class in python/carnot/pipeline/extract.py
- nsvif_tpr >= 0.50 AND nsvif_fpr <= 0.30 gate (relaxed for first prototype)

**exp2314** — VERGE SMT Repair v2 (codex, gated on exp2309.pretest_fixed)
- VERGERepair class: Minimal Correction Subset identification + targeted fix
- repair_violation_reduction_rate >= 0.60 gate
- Complement to NSVIF (detect-then-repair pipeline)

**exp2315** — Eidoku CSP Gate v3 (codex, gated on exp2309.pretest_fixed)
- EidokuCSPGate class: context-calibrated cost threshold
- TPR >= 0.60 AND FPR <= 0.20 gate
- Queued since .224; first actual run after pre-test fix

**exp2316** — Projected-Langevin vs CASAL v3 (codex, gated on exp2309.pretest_fixed)
- ProjectedLangevinSampler: exact constraint satisfaction vs CASAL
- Feasibility residual comparison on 3 problems (n=32)
- Queued since .224; first actual run

### Phase 3: Hardware Track + Adversarial Defence (exp2317–exp2319)

**exp2317** — KV260 RTL Verilator Lint v6 (codex, gated on exp2309.pretest_fixed)
- Verilator lint + Icarus 10-cycle simulation on parallel Ising RTL
- PRECONDITIONS: verilator + iverilog availability
- lint_errors_count == 0 gate for downstream synthesis claims

**exp2318** — ML-Assisted Ising Initialization (codex, gated on exp2309.pretest_fixed) — NEW
- arXiv:2503.23966: ML heuristic for better Ising initial states
- Implements lightweight initialization policy (gradient-free, CPU-only)
- Measures convergence speedup vs random initialization on n=64/128/256
- Sets ml_init_speedup_validated: true if speedup_ratio >= 1.5 on n=256

**exp2319** — Adversarial Null-Space Probe v5 (codex, gated on exp2309.pretest_fixed)
- k=16 ensemble adversarial input testing (200 inputs)
- ensemble_fooled_rate < 0.05 gate (validates AND-composition security)
- Queued since .224; first actual run after pre-test fix

### Phase 4: Capstone + Retro (exp2320–exp2321)

**exp2320** — Capstone E2E Live Generation (.227) (model: opus, max_turns: 100)
- Gated on exp2310.fst_live_validated AND exp2312.kancl_n256_validated
- 10 full verify-repair passes with SOTA GGUF, >50 tokens/answer
- capstone_passed: true if answers full-length AND repair rate >= 0.3

**exp2321** — Milestone .227 Retrospective (codex, max_turns: 20, ungated)
- Evaluates .226 retro gaps: pre-test cascade finally fixed?
- Top 3 successes and gaps for .228
- Updates ops/changelog.md

---

## Dependency Graph

```
exp2308 (archive) → exp2309 (pretest fix)
                         ↓ (pretest_fixed=true)
         ┌───────────────┼──────────────────────────────┐
    exp2310 (fst live)  exp2312 (kancl n256)            exp2313/2314/2315/2316/2317/2318/2319
         ↓                    ↓
    exp2311 (fr11)     [exp2320 capstone]
         ↑ ─────────────────┘
         exp2320 (capstone, model: opus)
    
exp2321 (retro, ungated)
```

---

## Hardware Requirements

- **Phase 1 (exp2310, exp2311, exp2320):** RTX 3090 CUDA, GGUF model cached in HuggingFace hub. PRECONDITIONS enforce model availability before any inference.
- **Phase 2 (exp2313, exp2314, exp2315, exp2316):** CPU-only (Z3 solver, Langevin dynamics). z3-solver Python package required (install if missing).
- **Phase 3 (exp2317):** Verilator + Icarus Verilog (toolchain check in PRECONDITIONS). exp2318 ML-Assisted Ising is CPU-only.
- **Phase 4 (exp2320):** RTX 3090 CUDA, GGUF model cached.

---

## FR-11 Mandate

**exp2311** satisfies the FR-11 continuous self-learning requirement for milestone .227 with `continuous_self_learning_task: true` in its artifact contract. Gate: `cross_domain_retention_rate >= 0.75`.

---

## Decentralization Check (Rules 1–7)

1. **Local-first:** All LLM tasks use locally-cached GGUF models (unsloth/Qwen3.6-35B-A3B-GGUF or gemma-4 variants). No closed-weight-only dependencies.
2. **Closed frontier optional:** NSVIF/VERGE use Z3 solver (open-source). No closed-weight API calls required.
3. **Distribution mirroring:** No new artifact publication in .227; this is a fix-and-run milestone.
4. **Multiple surfaces:** Experiments test Python API + CLI paths. MCP server compatibility maintained.
5. **Hardware portability:** ML-Assisted Ising (exp2318) runs on CPU, compatible with any substrate.
6. **Data minimization:** No external API calls in verification tasks.
7. **No vendor abstractions:** Z3, Verilator, llama.cpp are all open-source.

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` for all retired experiment scopes. Confirmed none of the following appear in .227 tasks:
- GRPO/VPRM v15 — not proposed
- WOPR puzzle cartridges — not proposed
- HardNet++/DSP — not proposed
- THRML scaling sweep — not proposed (vendored, no scaling sweep needed)
- SpecAnn — not proposed
- exp2091 (Tier 1 CSL Grammar) — not proposed
- iCE40 PIMI — not proposed (retired in .69/.70)

---

## Failed-Experiment Rerun Compliance

| Task | Prior Failure | Root Cause | What's Different |
|------|--------------|------------|-----------------|
| exp2309 | exp2295: partial_fix (pretest_fixed=false) | 2 specific test failures named: test_req_sample_041 + test_issue_14 | Target is exactly those 2 named tests with known fix path |
| exp2310 | exp2296: blocked_gate_check_failed | Upstream pretest_fixed=false (exp2295) | exp2309 achieves pretest_fixed=true first |
| exp2311 | exp2297: blocked_gate_check_failed | Upstream chain blocked | exp2309+exp2310 clear the gate chain |
| exp2312 | exp2298: blocked_gate_check_failed | Upstream pretest_fixed=false | exp2309 achieves pretest_fixed=true |
| exp2313 | (no prior run) | First implementation attempt; exp2301 blocked before run | exp2309 clears the pre-test blocker |
| exp2314 | (no prior run) | exp2302 blocked before run | exp2309 clears the pre-test blocker |
| exp2315 | exp2299: blocked_gate_check_failed | Pre-test SKIP despite no gated_on | exp2309 fixes the system-wide pre-test |
| exp2316 | exp2300: blocked_gate_check_failed | Same as exp2315 | Same |
| exp2317 | exp2303: blocked_gate_check_failed | Same | Same |
| exp2319 | exp2304: blocked_gate_check_failed | Same | Same |
| exp2320 | exp2306: blocked_gate_check_failed | Both upstream gates false | exp2310 + exp2312 clear both gates |
