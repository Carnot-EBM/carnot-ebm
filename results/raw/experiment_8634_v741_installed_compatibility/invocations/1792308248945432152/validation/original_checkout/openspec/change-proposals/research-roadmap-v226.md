# Research Roadmap — Milestone 2026.05.226

**Title:** Claude-Escalated Pre-Test Fix, FST Live Generation v6, NSVIF Neuro-Symbolic Verification, Eidoku CSP First Run, and Sparse Ising Connectivity

**Milestone:** 2026.05.226
**Date:** 2026-05-17
**Experiment IDs:** exp2294–exp2307 (14 tasks)
**Previous milestone:** 2026.05.225

---

## What Milestone 2026.05.225 Proved

Milestone .225 completed 3 of 14 tasks with gate-blocked artifacts and 11 tasks with no artifact, continuing the cascade pattern from .224:

1. **Archive step ran** (exp2280): research-roadmap.yaml now contains "2026.05.225" tasks; research-complete.yaml has .225 entries. The deliverable file is absent but the side effects succeeded — the conductor activated .225.

2. **Pre-test fix failed to produce a deliverable** (exp2281): `carnot.pypi_escalation` still lacks `check_pypi_escalation` and `run_escalation`. The codex gpt-5.5 agent (max_turns: 30) either got SKIP'd by the conductor's pre-test check or ran but failed to write a valid deliverable with `pretest_fixed: true`. This is the FIFTH consecutive milestone where the pre-test cascade has blocked substantive work.

3. **All gated tasks gate-blocked** (exp2282, exp2284, exp2286, exp2288, exp2292): No upstream deliverable was found, so all structured gates failed. Three of these produced gate-block artifacts; two have no files at all.

4. **Ungated tasks also did not run** (exp2289, exp2290, exp2291, exp2293): Despite having no `gated_on` field, these tasks produced no artifacts. The conductor's system-wide pre-test check SKIP'd them all before they could execute. This confirms: the conductor's pre-test failure blocks ALL tasks, not just gated ones.

5. **Root cause is STILL the same**: `tests/python/test_pypi_escalation.py` imports `check_pypi_escalation` and `run_escalation` from `carnot.pypi_escalation`, but the module only defines `generate_escalation_report` and `main`. The codex gpt-5.5 approach has been tried in exp2281 (.225) and is demonstrably insufficient. The next attempt uses Claude Sonnet (`requires_claude: true`) with C+E escalation to Opus, which has succeeded on harder multi-file Python debugging tasks in prior milestones (e.g., EnvPropagationGuard repair in .83).

**Key gaps entering .226:**

- Pre-test cascade (pypi_escalation missing symbols): 5th consecutive milestone affected. codex approach demonstrably failed — switching to Claude + Opus escalation.
- FST+ODAR+CASAL real-scale live generation: never achieved full-answer evidence (>50 tokens), blocked 5+ milestones.
- KAN-CL n=256 per-knot retention: blocked 5+ consecutive milestones by exogenous cascades.
- Eidoku CSP gate: never ran (was ungated but pre-test blocked in .225).
- Projected-Langevin baseline: never ran (same reason as Eidoku).
- Adversarial null-space probe: never ran (same reason).
- NSVIF neuro-symbolic verification: no prior attempt; first planned introduction.
- Sparse Ising connectivity: no prior attempt; first planned introduction.

---

## Architecture Snapshot

```
Milestone 2026.05.226 additions (bold = new this milestone):

Infrastructure:
  **carnot.pypi_escalation: add check_pypi_escalation + run_escalation** (exp2295)
  → Claude Sonnet + C+E Opus escalation replaces failed codex gpt-5.5 approach
  → 5th attempt at this fix; requires_claude: true with max_turns: 40

Verify-Repair Cascade:
  Tier 0a–0e probes → ODAR routing → Tier 1–2.7 → **Tier 2.8 Eidoku CSP** (exp2299) → Tier 3 Ising
                ^                                        ^
           **NSVIF Z3 extractor** (exp2301)    context-calibrated CSP gate
           arXiv:2601.17789                     smooth-falsehood rejection
           logical-step parser → Z3 prover       (first actual run)
           new python/carnot/pipeline/extract.py
           LogicalStepExtractor class

  After Tier 3 violation: **VERGE MCS repair** (exp2302)
  arXiv:2601.20055 — SMT Minimal Correction Subsets
  locates and repairs Z3-encodable violations post-detection

Constrained Sampling:
  CASAL (primal-dual) already implemented (.222)
  **Projected-Langevin sampler** (exp2300): first actual run
  ProjectedLangevinSampler comparison vs CASAL on 3 synthetic problems
  (was ungated in .225 but pre-test blocked; now gated on pretest_fixed)

Continuous Self-Learning (FR-11):
  FST slow/fast weight decomposition → **multi-domain retention v3** (exp2297)
  Gated on exp2296.fst_live_validated (ensures live generation works first)
  cross_domain_retention_rate >= 0.75 closes FR-11 multi-domain claim

KAN Tier:
  KAN-CL per-knot importance → **n=256 validation v5** (exp2298)
  n256_retention_rate >= 0.85 closes the 5-milestone carry-forward

Hardware Track (active):
  KV260 RTL → **Verilator lint + Icarus sim v5** (exp2303, retry)
  Yosys synthesis already attempted; sparse Ising connectivity (exp2305)
  arXiv:2503.01177 — copy-node graph sparsification → LUT reduction

Defence Layer (Phase 3):
  k=16 ensemble → **adversarial null-space probe v4** (exp2304, retry)
  ensemble_fooled_rate < 0.05 validates AND-composition security property
```

---

## The 3 Biggest Gaps vs PRD Vision

1. **Pre-test cascade blocks all meaningful work (5th consecutive milestone)** — The fundamental blocker is unchanged: `tests/python/test_pypi_escalation.py` imports two functions that don't exist in `carnot.pypi_escalation`. The codex gpt-5.5 approach has demonstrably failed across two milestones (.224 exp2267 fixed the DualGPU issue but missed this; .225 exp2281 was aimed directly at this and produced no deliverable). The fix for .226 uses Claude Sonnet with `requires_claude: true` and the C+E Opus escalation path, which succeeded on equivalent multi-file Python debugging tasks in milestone .83 (EnvPropagationGuard repair).

2. **FR-11 continuous self-learning has no live-model evidence** — FR-11 (autonomous self-learning loop, PRD section) requires the verifier-ensemble learning mechanism to close a real feedback loop on live LLM outputs. FST multi-domain retention (exp2297) is the current milestone's FR-11 task. But it requires live generation evidence first (exp2296). After 5+ milestones of cascade blocking, the live generation evidence has never been captured. exp2295 → exp2296 → exp2297 is the critical path.

3. **Neuro-symbolic extraction not yet implemented (PRD priority #1)** — Research-program.md lists "Rebuild constraint extraction for real models" as the HIGHEST PRIORITY since 2026-04-11. The NSVIF approach (arXiv:2601.17789) was identified months ago as the target implementation. It has never been attempted in any experiment. exp2301 (NSVIF implementation) is the first direct attempt.

---

## Phase Structure

### Phase 0: Infrastructure (2 tasks, no gate dependencies)

| Task | Scope | Agent | Gate produced |
|------|-------|-------|---------------|
| exp2294 | Archive .225, activate .226 | codex gpt-5.5 | `archive_ready: true` |
| exp2295 | Fix `carnot.pypi_escalation` missing symbols (requires_claude: true) | Claude Sonnet + Opus escalation | `pretest_fixed: true` |

### Phase 1: Core Experiments (3 tasks, gated on exp2295.pretest_fixed)

| Task | Scope | Agent | Gate produced |
|------|-------|-------|---------------|
| exp2296 | FST+ODAR+CASAL real-scale live gen v6 (>50 tokens) | codex gpt-5.5 | `fst_live_validated: true` |
| exp2297 | FR-11 FST multi-domain retention v3 | codex gpt-5.5 | `fr11_multidomain_passed: true` |
| exp2298 | KAN-CL n=256 per-knot retention v5 | codex gpt-5.5 | `kancl_n256_validated: true` |

### Phase 2: New Research (4 tasks, gated on exp2295.pretest_fixed)

| Task | Scope | Agent | Gate produced |
|------|-------|-------|---------------|
| exp2299 | Eidoku CSP Tier 2.8 gate (first actual run) | codex gpt-5.5 | `eidoku_gate_validated: true` |
| exp2300 | Projected-Langevin vs CASAL baseline (first actual run) | codex gpt-5.5 | `projected_langevin_competitive: true` |
| exp2301 | NSVIF neuro-symbolic Z3 extractor (NEW, arXiv:2601.17789) | codex gpt-5.5 | `nsvif_extractor_validated: true` |
| exp2302 | VERGE SMT repair integration (NEW, arXiv:2601.20055) | codex gpt-5.5 | `verge_repair_validated: true` |

### Phase 3: Hardware + Defence (3 tasks, gated on exp2295.pretest_fixed)

| Task | Scope | Agent | Gate produced |
|------|-------|-------|---------------|
| exp2303 | KV260 RTL Verilator lint v5 | codex gpt-5.5 | `lint_errors_count: 0` |
| exp2304 | Adversarial null-space probe v4 | codex gpt-5.5 | `adversarial_probe_passed: true` |
| exp2305 | Sparse Ising connectivity (NEW, arXiv:2503.01177) | codex gpt-5.5 | `sparse_lut_reduction_pct: float` |

### Phase 4: Synthesis + Retro (2 tasks)

| Task | Scope | Agent | Gate |
|------|-------|-------|------|
| exp2306 | Capstone E2E live gen (.226) — FST + ODAR + CASAL + KAN-CL n=256 | Claude Opus | gated on exp2296.fst_live_validated AND exp2298.kancl_n256_validated |
| exp2307 | Milestone 2026.05.226 Retrospective | codex gpt-5.5 | ungated (always runs) |

---

## Dependency Graph

```
exp2294 (archive)
    |
exp2295 (pretest fix — Claude + Opus escalation)
    |
    +——> exp2296 (FST live gen v6) ——> exp2297 (FR-11 multidomain v3)
    |                                       |
    +——> exp2298 (KAN-CL n=256 v5)         |
    |         |                             |
    +——> exp2299 (Eidoku CSP v2)           |
    |                                       +——> exp2306 (capstone Opus)
    +——> exp2300 (Projected-Langevin v2)       (gated on 2296+2298)
    |
    +——> exp2301 (NSVIF neuro-symbolic NEW)
    |
    +——> exp2302 (VERGE repair NEW)
    |
    +——> exp2303 (KV260 RTL lint v5)
    |
    +——> exp2304 (adversarial probe v4)
    |
    +——> exp2305 (sparse Ising NEW)

exp2307 (retro — ungated, always runs last)
```

---

## Failed-Experiment Rerun Compliance

Per CLAUDE.md Failed-Experiment Rerun Discipline, every carry-forward experiment must name the prior failure, root cause, and what is different:

| Task | Prior failure chain | Diagnosed root cause | What is different in .226 |
|------|---------------------|---------------------|--------------------------|
| exp2295 | exp2281 (.225 codex), exp2267 (.224) | codex gpt-5.5 fails to produce pretest_fixed=true deliverable | requires_claude: true, max_turns: 40, C+E Opus escalation |
| exp2296 | exp2282 (.225), exp2268 (.224), exp2255 (.223) | upstream pretest cascade | exp2295 fixes pre-test first |
| exp2297 | exp2283 (.225), exp2269 (.224), exp2256 (.223) | exp2296 gate never opened | exp2296 now runs after pre-test fix |
| exp2298 | exp2284 (.225), exp2270 (.224), exp2258 (.223), exp2247 (.222) | upstream pretest cascade | exp2295 fixes pre-test first |
| exp2299 | exp2289 (.225, not_run) | ungated but pre-test SKIP'd system-wide | now gated on pretest_fixed; pre-test fixed by exp2295 |
| exp2300 | exp2290 (.225, not_run) | same as exp2299 | same fix |
| exp2303 | exp2286 (.225), exp2272 (.224), exp2260 (.223), exp2249 (.222) | upstream pretest cascade | exp2295 fixes pre-test first |
| exp2304 | exp2288 (.225), exp2274 (.224), exp2262 (.223) | upstream pretest cascade (plus ungated SKIP) | exp2295 fixes pre-test first |

New experiments (first run, no prior failure required): exp2301, exp2302, exp2305.

---

## Exclusion Manifest Cross-Check

Per CLAUDE.md Exclusion-Manifest Cross-Check Before Planning, the following retired scopes are confirmed NOT proposed in this milestone:

- GRPO/VPRM lineage (retired .112): not proposed
- WOPR puzzle cartridges (retired .112): not proposed
- HardNet++/DSP repair stack (retired .112): not proposed
- THRML scaling sweep (retired .120): not proposed
- SpecAnn Phase 3 sampler (retired .120): not proposed
- exp2091 Gemini CSL Grammar (retired .164): not proposed

---

## Hardware Requirements

| Track | Hardware needed | Status |
|-------|----------------|--------|
| FST+ODAR+CASAL live gen | RTX 3090 CUDA (GGUF inference) | Available; must pass PRECONDITIONS check |
| KAN-CL n=256 | CPU only (JAX) | Always available |
| Eidoku CSP | CPU only | Always available |
| Projected-Langevin | CPU only (JAX) | Always available |
| NSVIF Z3 extractor | CPU only (Z3 solver) | Needs `pip install z3-solver` check |
| VERGE repair | CPU only (Z3 solver) | Needs `pip install z3-solver` check |
| KV260 RTL lint | CPU (Verilator + Icarus) | Available if tools present |
| Sparse Ising | CPU (Python) | Always available |
| Adversarial probe | CPU or GPU | Available |
| Capstone | RTX 3090 CUDA (GGUF inference) | Available; gated on exp2296 |

---

## Decentralization Check

Per CLAUDE.md Decentralization-Respecting Design Constraints:

- **Rule 1 (local-first)**: All LLM-bearing experiments use local GGUF models (Qwen3.6-35B, gemma-4-31B, gemma-4-26B). NSVIF uses Z3 (open-source). VERGE uses Z3. No closed-weight API required.
- **Rule 2 (closed models optional)**: exp2296 and exp2306 have PRECONDITIONS checks that emit `blocked_model_not_cached` if no SOTA GGUF is available, rather than failing silently or calling cloud APIs.
- **Rule 3 (distribution mirroring)**: No new artifact publication in this milestone; existing PyPI + IPFS policy unchanged.
- **Rule 4 (multiple integration surfaces)**: Python API, CLI, MCP server all remain available; no surface is locked out.
- **Rule 5 (hardware portability)**: KV260 RTL + sparse Ising connectivity advance the sovereignty hardware story.
- **Rule 6 (data minimization)**: No closed-weight LLM calls in any experiment.
- **Rule 7 (no vendor abstractions in core)**: NSVIF and VERGE use Z3 (open-source); implementations go in pipeline/extract.py and verify/ respectively, behind abstract protocols.

---

## FR-11 Continuous Self-Learning Mandate

**FR-11 task**: exp2297 (FR-11 FST Multi-Domain Retention v3)
- `continuous_self_learning_task: true` in artifact contract
- Gate: `cross_domain_retention_rate >= 0.75`
- Gated on: exp2296.fst_live_validated == true
- Domain sequence: math → code → logic (10 examples each)
- Measures: cross-domain retention after sequential training

---

## What Success Looks Like for .226

**Minimum viable outcome** (pre-test fix only):
- exp2295 produces `pretest_fixed: true`
- At least one downstream experiment runs and closes its gate

**Full success**:
- exp2295 `pretest_fixed: true` → all Phase 1-3 experiments unlock
- exp2296 `fst_live_validated: true` (live generation >50 tokens confirmed)
- exp2298 `kancl_n256_validated: true` (5-milestone carry-forward closed)
- exp2299 `eidoku_gate_validated: true` (Tier 2.8 first validated)
- exp2301 `nsvif_extractor_validated: true` (PRD priority #1 first prototype)
- exp2306 capstone passes with full-answer generation

**Fallback** (if pre-test fix fails again):
- exp2307 (retro) documents the failure
- exp2295 scope escalated to `model: opus` with `operator_override` for .227
