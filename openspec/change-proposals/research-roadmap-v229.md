# Research Roadmap — Milestone 2026.05.229

**Prepared:** 2026-05-18
**Milestone:** 2026.05.229
**Experiment IDs:** exp2336 – exp2349
**Previous milestone:** 2026.05.228 (0/14 criteria met; 8th consecutive empty-experiment milestone; pre-test cascade still unresolved)

---

## What Milestone .228 Proved

Milestone .228 produced a 14-task roadmap (exp2322–exp2335) but completed 0 research experiments.
The operational retro (results/operational_retro_2026_05_228.json) records:

- **exp2322 (archive):** Ran; archived .227 to research-complete.yaml. Clean.
- **exp2323 (pre-test fix v9, requires_claude:true):** Did not produce a `pretest_fixed: true` artifact
  within the milestone window. Root causes remain fully diagnosed (3 specific fixes) but unexecuted.
- **exp2324–exp2333 (Phase 1–3):** All GATE_BLOCK because exp2323 never set `pretest_fixed: true`.
- **exp2334 (capstone):** GATE_BLOCK (upstream gates never opened).
- **exp2335 (retro):** Ran; confirmed 0 experiments, 0 wall-time, both GPUs idle.

**Systemic finding:** After 8 consecutive milestones with 0 experiments, the failure is a
PROCESS failure, not a code failure. The conductor generates the roadmap but does not execute
exp2323 within the same milestone window (activation latency). The structural fix is to:
(a) guarantee exp2337 (pre-test fix v10) is ungated and executes immediately on activation,
(b) add ungated research tasks (exp2338) so the milestone produces results regardless of
    whether pre-test is fixed, and
(c) document the operator manual intervention path if attempt 10 also fails.

---

## Three Biggest Gaps vs PRD Vision

### Gap 1: Pre-test cascade blocking all research (9 consecutive attempts, 8 empty milestones)

**Root cause is fully diagnosed** (from exp2309 + exp2323 analysis):
1. `results/experiment_1692_potts_export.json` missing — `test_experiment_1692_potts_v2.py` requires it
2. `test_experiment_390_gpu_preflight.py::TestRunGpuPreflight` passes in isolation, fails under xdist
   GPU contention — fix is `@pytest.mark.xdist_group("gpu_serial")` on the class
3. `test_experiment_294_gpu_baseline_apple.py::TestBaselineAccuracyBounds` errors under xdist memory
   leak — fix is the same xdist group marker

These 3 fixes are CONCRETE and TARGETED. They have not been applied because every previous
attempt (exp2267/2281/2295/2309) either timed out or was never activated within the milestone window.

**New structural change for .229:** exp2337 is UNGATED (no `gated_on` block). It will run
immediately after the archive step regardless of any upstream gate state. This is the primary
change from .228's exp2323, which was also ungated in specification but may have failed to activate
due to conductor timing.

**Operator escalation path (if attempt 10 fails):** exp2337's prompt includes explicit commands
for the operator to run manually in a terminal if the Claude-escalated fix still fails. This
converts a cascade blocker into an operator action item with a 5-minute manual fix path.

### Gap 2: NSVIF neuro-symbolic extraction never ran (PRD Priority #1 since 2026-04-11)

`research-program.md` lists "Rebuild constraint extraction for real models" as the HIGHEST
PRIORITY. The ArithmeticExtractor found ZERO violations on instruction-tuned models. NSVIF
(arXiv:2601.17789) — the Z3-based neuro-symbolic replacement — has been proposed for 4
consecutive milestones (.225, .226, .227, .228) but has never executed.

Once exp2337 clears the pre-test gate, exp2342 (NSVIF v4) runs immediately. If exp2337
fails for a 10th time, exp2338 (Semantic Energy, ungated) still validates the hallucination
detection thesis using a different mechanism while NSVIF waits.

### Gap 3: FST live generation validated at token-level only

The FST+ODAR+CASAL pipeline stack has been blocked for 8 milestones. The Phase 3/4 capstone
depends on both `fst_live_validated` (full answer generation, not single-token probe) and
`kancl_n256_validated`. Once the pre-test cascade clears, these run in Phase 1.

---

## Architecture Snapshot (2026-05-18)

```
                    [ LLM Inference ]
                    (SOTA GGUF: Qwen3.6-35B, Gemma4-31B, Gemma4-26B)
                           |
                    [ ODAR Router ]         <- arXiv:2602.23681
                   /              \
            [Fast Path]      [Deliberative]
           (Tier 0 probes)         |
                         [ FST Context Prep ]  <- arXiv:2605.12484
                                  |
                         [ CASAL Sampler ]      <- arXiv:2603.07204
                                  |
                    [ Verifier Ensemble k=16 ]
                    +-- Boolean E (Ising)
                    +-- Z3/NSVIF            <- arXiv:2601.17789 [NOT YET RUN]
                    +-- VERGE MCS Repair    <- arXiv:2601.20055 [NOT YET RUN]
                    +-- Eidoku CSP          <- arXiv:2512.20664 [NOT YET RUN]
                    +-- KAN-CL n=256        <- arXiv:2605.12306 [BLOCKED]
                    +-- Semantic Energy     <- arXiv:2508.14496 [NEW in .229]
                    +-- SpilledEnergy, HalluField, FAA
                           |
                    [ Repair Output ]
                           |
              [ Hardware: KV260 FPGA / RTL Lint ]
              [ ML-Assisted Init: arXiv:2503.23966 ]
```

**Ungated in .229:** exp2338 (Semantic Energy Tier 0g) runs without the pre-test gate.
This ensures at least one new verifier research result lands regardless of cascade state.

---

## Phase Structure

### Phase 0: Archive + Pre-Test Fix (ALWAYS RUNS)

**exp2336** — Archive .228, activate .229 (codex, max_turns:20, UNGATED)
- Archive exp2322–exp2335 to research-complete.yaml
- Replace research-roadmap.yaml with research-roadmap-next.yaml
- Add changelog stub

**exp2337** — Pre-Test Cascade Fix v10 (requires_claude:true, max_turns:50, UNGATED)
- 10th attempt. Root causes fully diagnosed in exp2309/exp2323.
- Three specific fixes: potts artifact + xdist group markers x2
- If still failing: write exact operator commands for manual terminal intervention
- **What's different from exp2323:** Explicit operator escalation path + documentation
  of manual fix commands in the artifact. If the suite is still failing after all 3
  targeted fixes, exp2337 emits `tests_still_failing` with exact pytest commands so
  the operator can reproduce in a terminal in < 5 minutes.

### Phase 1: Ungated New Research (ALWAYS RUNS — does NOT gate on pre-test)

**exp2338** — Semantic Energy Hallucination Detector (arXiv:2508.14496) (codex, max_turns:30, UNGATED)
- Implements Boltzmann-energy hallucination detector on penultimate layer logits
- New Tier 0g verifier: orthogonal to SpilledEnergy (Tier 0b) and FAA (Tier 0f)
- Pure Python, no GPU needed, no test suite dependency
- Gate: `semantic_energy_auroc >= 0.60` on synthetic violation corpus

### Phase 2: Research Retries (ALL gated on exp2337.pretest_fixed == true)

**exp2339** — FST+ODAR+CASAL Real-Scale Live Generation v9 (codex, max_turns:50)
- 9th attempt. Blocked 8 milestones by cascade.
- What's different: gate now points to exp2337 (ungated fix); model cache PRECONDITIONS checked first.

**exp2340** — FR-11 FST Multi-Domain Retention v6 (codex, max_turns:30)
- 6th attempt. Gated on exp2337.pretest_fixed AND exp2339.fst_live_validated.
- Continuous self-learning mandate (FR-11 required per milestone).

**exp2341** — KAN-CL n=256 Per-Knot Retention v8 (codex, max_turns:30)
- 8th attempt. Gated on exp2337.pretest_fixed.

**exp2342** — NSVIF Neuro-Symbolic Z3 Extractor v4 (codex, max_turns:40)
- 4th attempt. PRD Priority #1 since 2026-04-11. Gated on exp2337.pretest_fixed.

**exp2343** — VERGE SMT Minimal Correction Subset Repair v4 (codex, max_turns:30)
- 4th attempt. Gated on exp2337.pretest_fixed.

**exp2344** — Eidoku CSP Tier 2.8 Gate v5 (codex, max_turns:30)
- 5th attempt. Gated on exp2337.pretest_fixed.

**exp2345** — Projected-Langevin vs CASAL Baseline v5 (codex, max_turns:30)
- 5th attempt. Gated on exp2337.pretest_fixed.

### Phase 3: Hardware + Adversarial (gated on pretest_fixed)

**exp2346** — KV260 RTL Verilator Lint + Icarus Simulation v8 (codex, max_turns:30)
- 8th attempt. Gated on exp2337.pretest_fixed.

**exp2347** — ML-Assisted Ising Machine Initialization v3 (codex, max_turns:30)
- 3rd attempt. Gated on exp2337.pretest_fixed.

### Phase 4: Capstone + Retro

**exp2348** — Capstone E2E Live Generation .229 (opus, max_turns:100)
- Gated on exp2339.fst_live_validated AND exp2341.kancl_n256_validated.

**exp2349** — Milestone 2026.05.229 Retrospective (codex, max_turns:20, UNGATED)
- Always runs last.

---

## Dependency Graph

```
exp2336 (archive, ungated)
  |
exp2337 (pre-test fix, ungated, requires_claude)
  |                                    \
  |                              exp2338 (Semantic Energy, UNGATED)
  +---> exp2339 (FST live v9) ----> exp2340 (FR-11 multidomain)
  |                                        |
  +---> exp2341 (KAN-CL n=256) -------+---+
  |                                   |
  +---> exp2342 (NSVIF v4)           +---> exp2348 (capstone, opus)
  +---> exp2343 (VERGE v4)           |
  +---> exp2344 (Eidoku CSP v5)      |
  +---> exp2345 (Proj-Langevin v5)   |
  +---> exp2346 (KV260 RTL v8)       |
  +---> exp2347 (ML Ising Init v3)   |
                                      |
exp2349 (retro, ungated) <-----------+
```

---

## FR-11 Continuous Self-Learning Mandate

**exp2340** carries `continuous_self_learning_task: true` with gate
`cross_domain_retention_rate >= 0.75`. FST fast-weight updates must generalize
across 3 reasoning domains (math, code, logic) — this is the FR-11 mandate.

---

## Hardware Requirements

- **exp2339, exp2348:** SOTA GGUF cached (`~/.cache/huggingface/hub/models--unsloth--*`).
  Minimum one of: Qwen3.6-35B-A3B-GGUF, gemma-4-31B-it-GGUF, gemma-4-26B-A4B-it-GGUF.
- **exp2346:** Verilator + Icarus (`command -v verilator && command -v iverilog`).
- **exp2347:** CPU only (gradient-free MLIsingInitializer).
- **exp2338:** CPU only (penultimate layer logit analysis, no GPU required).
- All others: CPU + Python environment.

---

## Decentralization Check (CLAUDE.md Rules 1–7)

| Rule | Compliant? | Notes |
|------|-----------|-------|
| 1. Local-first open models | YES | exp2339/2348 use mandated local GGUF models |
| 2. Closed models optional | YES | No closed-model dependency in any experiment |
| 3. Distribution mirroring | N/A | No new weights published this milestone |
| 4. Multiple integration surfaces | YES | Python API + MCP server maintained |
| 5. Hardware portability | YES | CPU-only paths for all non-GGUF experiments |
| 6. Per-call data minimization | YES | No closed-weight calls |
| 7. No vendor abstractions in core | YES | All new modules in `carnot.verify.`, `carnot.samplers.` |

---

## Exclusion Manifest Cross-Check

Retired scopes checked against all proposed tasks:

| Retired scope | Match in .229? | Status |
|---|---|---|
| exp2091 (gemini CLI bail-out) | No gemini tasks | CLEAR |
| GRPO/VPRM v1-v14 | No GRPO/VPRM | CLEAR |
| WOPR puzzle cartridges | No WOPR | CLEAR |
| HardNet++/DSP | No HardNet++ | CLEAR |
| THRML scaling sweep | No THRML parity tasks | CLEAR |
| SpecAnn | No Spectral Annealing | CLEAR |
| iCE40 PIMI | No PIMI tasks | CLEAR |

All clear. No proposed tasks match any retired experiment scope.

---

## Failed-Experiment Rerun Compliance

| Task | Prior failures | Root cause | What's different | Retire if same? |
|------|---------------|------------|------------------|-----------------|
| exp2337 (pretest v10) | exp2323 (v9, never ran), exp2309 (v8, timed out), exp2295 (v7), exp2281 (v6), exp2267 (v5) | Conductor never activates within milestone window; 3 remaining test targets fully diagnosed | UNGATED execution + operator manual intervention path documented in artifact | YES |
| exp2338 (Semantic Energy) | None — NEW | N/A | N/A | N/A |
| exp2339 (FST v9) | exp2324 (v8), exp2310 (v7), exp2296 (v6) | Gated on pretest_fixed which was never true | Gate points to exp2337 (ungated) | YES |
| exp2340 (FR-11 v6) | exp2325 (v5), exp2311 (v4), exp2297 (v3) | Same cascade | Same fix | YES |
| exp2341 (KAN-CL v8) | exp2326 (v7), exp2312 (v6), exp2298 (v5) | Same cascade | Same fix | YES |
| exp2342 (NSVIF v4) | exp2327 (v3), exp2313 (v2), exp2301 (v1) | Same cascade | Same fix | YES |
| exp2343 (VERGE v4) | exp2328 (v3), exp2314 (v2), exp2302 (v1) | Same cascade | Same fix | YES |
| exp2344 (Eidoku v5) | exp2329 (v4), exp2315 (v3), exp2299 (v2) | Same cascade | Same fix | YES |
| exp2345 (Proj-Langevin v5) | exp2330 (v4), exp2316 (v3), exp2300 (v2) | Same cascade | Same fix | YES |
| exp2346 (KV260 v8) | exp2331 (v7), exp2317 (v6), exp2303 (v5) | Same cascade | Same fix | YES |
| exp2347 (ML Ising v3) | exp2332 (v2), exp2318 (v1) | Same cascade | Same fix | YES |
| exp2348 (capstone v229) | exp2334 (v228), exp2320 (v227) | Same upstream gates never true | Same fix | YES |
| exp2349 (retro) | exp2335 (v228, complete) | Retros always run — no failure | New instance | YES |
