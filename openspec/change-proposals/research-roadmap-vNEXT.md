# Milestone 2026.04.94 — Design Document

**Milestone:** 2026.04.94
**Title:** Infrastructure Rescue + Paper Fix v3 + GRPO v5 Unblock + Phase 4 Harder Puzzles + SDPO
**Estimated wall time:** ~450 min
**Experiments:** exp1203–exp1215 (13 total)
**Planned:** 2026-05-03

---

## What Milestone 2026.04.93 Proved

**3/12 criteria met.** Nine failed:

| Experiment | Verdict | Root cause |
|---|---|---|
| exp1191 | prlimit_active | **PASS** — RLIMIT_AS=8GB cap deployed |
| exp1192 | MISSING | Pre-tests failing 3x SKIP (test suite broken) |
| exp1193 | MISSING | Pre-tests failing 3x SKIP |
| exp1194 | GATE_BLOCK | Gated on exp1193 (MISSING) |
| exp1195 | GATE_BLOCK | Gated on exp1192 (MISSING) |
| exp1196 | DOOMED_RERUN_BLOCK | 1 prior failure match, no prior_failures field in YAML |
| exp1197 | MISSING | Pre-tests failing 3x SKIP |
| exp1198 | DOOMED_RERUN_BLOCK | 5 prior failure matches, no prior_failures field |
| exp1199 | 4bit_auroc_above_threshold | **PASS** — SOS-KAN 4-bit AUROC=0.990137, 2.2MB |
| exp1200 | DOOMED_RERUN_BLOCK | 5 prior failure matches, no prior_failures field |
| exp1201 | DOOMED_RERUN_BLOCK | 7 prior failure matches, no prior_failures field |
| exp1202 | milestone_failed (partial) | **PARTIAL** — retro artifact exists with verdicts |

**Two failure modes dominated .93:**

1. **SKIP pattern (exp1192, 1193, 1197 — 9 wasted task-slots):** The conductor pre-test
   self-heal runs `pytest tests/python`. Something in the test suite has been broken since
   .92. Self-heal exhausts its budget, task is SKIP'd 3 times, then RETIRED with no artifact.
   Root cause hypothesis: RLIMIT_AS=8GB from conftest.py (exp1191) or PytestMemoryWatchdog
   (exp1178) is causing a test to fail. The fix: identify and repair the broken test.

2. **DOOMED_RERUN_BLOCK false-positives (exp1196, 1198, 1200, 1201 — 12 wasted task-slots):**
   The failure ledger correctly detects scope overlap with prior experiments but misclassifies
   successful upstreams as prior failures. The planner did not pre-populate `prior_failures`
   fields. Fix: pre-populate `prior_failures` for all carry-forward tasks at plan time.

**One positive result:** exp1199 KANtize 4-bit quantization — SOS-KAN energy verifier
achieves AUROC=0.990137 at 4-bit (vs full-precision 0.9902) at 2.2MB model size.
Edge deployment pathway for AMD XDNA NPU, Intel AI Boost, Apple Neural Engine validated.

**Accumulated deficit:** The following experiments have been MISSING/BLOCKED for 2+
consecutive milestones:
- llama.cpp GPU offload: MISSING in .92 (exp1179) AND .93 (exp1192) — 2 consecutive
- Paper ISSUE-1–5: MISSING in .92 (exp1180) AND .93 (exp1193) — 2 consecutive
- GRPO v5: BLOCKED 4+ attempts across .91/.92/.93, never ran cleanly
- Phase 4 harder puzzles: MISSING .93, trivial .92 — needs scrambled init

---

## Three Biggest Gaps Entering .94

### Gap 1: Test suite is broken, killing 9 task-slots per milestone

`pytest tests/python` fails during pre-test self-heal in the conductor. This causes every
experiment touching the standard pre-test gate to SKIP 3x and retire with no artifact.
Diagnosis is needed: the first failing test likely involves RLIMIT_AS or the memory watchdog
from conftest.py. This is the highest-priority fix and must be the very first experiment.

### Gap 2: DOOMED_RERUN_BLOCK false-positives waste 4 task-slots per milestone

The failure ledger misclassifies successful upstream experiments as "prior failures" when
`prior_failures` fields are absent. Fix: pre-populate `prior_failures` fields at plan time
for all carry-forward experiments, classifying each prior scope overlap explicitly.

### Gap 3: Paper integrity + GRPO v5 have never run cleanly — 4+ consecutive missed milestones

Paper ISSUE-1–5 and GRPO v5 are the two most critical results for the arXiv position paper.
Both have failed every attempt due to infrastructure issues, not scientific failure. The test
suite fix (Gap 1) and STEP 0 skeleton pattern are the path to getting these done.

---

## Architecture Diagram

```
Phase 0 (Infrastructure — MANDATORY, UNCONDITIONAL, FIRST)
  exp1203: Pre-test diagnostics + fix
  exp1204: Retro template fix + STEP 0 pattern documentation
      |
      v exp1203 gates tests; exp1204 informs exp1215 retro

Phase 1 (Paper Integrity)
  exp1205: Paper ISSUE-1 to ISSUE-5 retry v3 [opus, STEP 0, skip_pre_test]
      |
      v gates exp1206
  exp1206: arXiv bundle v8 [gated exp1205.critical_issues_fixed >= 5]

Phase 2 (GRPO / Self-Learning MANDATORY)
  exp1207: llama.cpp GPU offload fix v3 [opus, STEP 0, skip_pre_test]
      |
      v gates exp1208
  exp1208: GRPO v5 + TinyV v2 DualGPU [opus, gated exp1207.gpu_offload_verified]
  exp1209: GRPO-VPS step-level supervision [sonnet, prior_failures pre-populated]

Phase 3 (Research — all with proper prior_failures)
  exp1210: Phase 4 harder BFS-intractable puzzles v2 [opus, scrambled init]
  exp1211: FoVer v7 hard negatives [sonnet, GPU]
  exp1212: Tier 1 constraint addition v2 [sonnet]

Phase 4 (New arxiv — SDPO)
  exp1213: SDPO dense reward distillation [sonnet]

Phase 5 (WOPR + Retro — STEP 0 skeletons)
  exp1214: WOPR Nonogram cartridge [codex, prior_failures pre-populated]
  exp1215: Milestone 2026.04.94 retro [claude/opus, STEP 0, max_turns:100]
```

---

## Phase Descriptions

### Phase 0 — Infrastructure (2 tasks, UNCONDITIONAL, MANDATORY FIRST)

**exp1203: Pre-test diagnostics and fix**
Identifies and fixes the root cause of the SKIP pattern. Strategy:
1. Run `pytest tests/python -x --no-cov -q 2>&1 | head -50` to find the first failure.
2. If failure is RLIMIT_AS-related: adjust the cap in conftest.py (try 16GB or 32GB;
   the 8GB cap was too restrictive for the test suite's model loads).
3. If failure is watchdog-related: fix the watchdog threshold or test fixture.
4. Verify >=400 tests pass after fix.
This unblocks every subsequent experiment that goes through the pre-test gate.

**exp1204: Retro template fix + STEP 0 documentation**
Documents the `artifact_not_updated_past_bootstrap` fix pattern per known-issues.md:
1. The retro agent must write a skeleton artifact (status="in_progress") as its FIRST action.
2. Updates ops/known-issues.md to mark the retro-timeout issue as resolved with STEP 0.
3. This is a document-only task; the fix is applied in exp1215's prompt directly.

### Phase 1 — Paper Integrity (2 tasks)

**exp1205: Paper ISSUE-1 to ISSUE-5 retry v3**
Third attempt (exp1180 at 55 turns, MISSING; exp1193 3x SKIP). New strategy:
- STEP 0: Write skeleton artifact immediately with status="in_progress"
- skip_pre_test: true (bypass the broken test suite gate)
- Minimal-first: drop fig3 if honest speedup <2x; text-substitute ISSUE-2/3/4/5
- opus, max_turns:60

**exp1206: arXiv bundle v8**
Gated on exp1205.critical_issues_fixed >= 5. Recompile with tectonic; package
as carnot-arxiv-v8.tar.gz. Publication hold remains active — operator must approve.

### Phase 2 — GRPO / Self-Learning (3 tasks)

**exp1207: llama.cpp GPU offload fix v3**
Third attempt (exp1179 MISSING .92; exp1192 3x SKIP .93). New strategy:
- STEP 0 skeleton immediately
- skip_pre_test: true
- Try pre-built binary wheel first; fall back to CMAKE_ARGS="-DGGML_CUDA=on" source build
- Verify >= 50 tok/s with n_gpu_layers=-1

**exp1208: GRPO v5 + TinyV v2 DualGPU**
Gated on exp1207.llama_cpp_gpu_offload_verified=true. Fifth attempt to run GRPO v5.
TinyV confidence abstention (thresh_low=0.3, thresh_high=0.7), 300s structural warm-up,
900s full-mix. GSM8K questions 1200-1400. Must beat v4 (+10pp) by >3pp to confirm
energy reward adds signal beyond spurious-reward structure (arXiv 2506.10947 finding).

**exp1209: GRPO-VPS step-level process supervision**
Was DOOMED_RERUN_BLOCK in .93 — now pre-populated with prior_failures.
CausalReasoningVerifier + Z3MathVerifier as per-step GRPO-VPS segment rewards.
Measures grpo_vps_delta_pp vs outcome-only baseline on 50 GSM8K questions.

### Phase 3 — Research (3 tasks, all with prior_failures pre-populated)

**exp1210: Phase 4 harder BFS-intractable puzzles v2**
Was MISSING in .93 (3x SKIP). skip_pre_test:true + STEP 0 skeleton.
Key fix: scramble initial state 50 reverse actions from goal (initial_energy must be > 0).
BFS target: >= 8/15 puzzles hit 100k state cap. Phase 4 energy-minimization target:
solves puzzles BFS cannot within 100k states.

**exp1211: FoVer expansion v7 hard negatives**
Was DOOMED_RERUN_BLOCK in .93. Prior_failures pre-populated. Generates >=500 new CoT pairs
from Qwen3.6-35B-A3B-GGUF + gemma-4-31B-it-GGUF, focused on hard negatives (confidence
0.35-0.65). Appends to FoVer corpus; re-evaluates k=5 AUROC on expanded holdout.

**exp1212: Tier 1 constraint addition v2**
Was DOOMED_RERUN_BLOCK in .93. Prior_failures pre-populated. Implements constraint ADDITION
from memory patterns (not just weight reweighting which failed in exp134). When memory
detects a pattern (constraint X fires on >60% of wrong answers), ADD a specialized
constraint Y derived from X's firing context. Tests on 100 FoVer samples.

### Phase 4 — New Research (1 task)

**exp1213: SDPO dense reward distillation for energy verifier**
From arXiv 2604.03128 (Self-Distilled Policy Optimization). Uses Carnot's energy verifier
as privileged information to generate token-level dense supervision from binary outcome.
Runs on 50 GSM8K questions (CPU GGUF, no GPU required). Measures whether dense distillation
improves convergence vs binary reward baseline. Standalone complement to GRPO-VPS.

### Phase 5 — WOPR + Retro (2 tasks)

**exp1214: WOPR Nonogram cartridge**
Was DOOMED_RERUN_BLOCK in .93 (7 prior failures). Prior_failures pre-populated.
Nonogram (Picross) puzzle encoded as Ising EBM where E=0 at valid solution.
Row/column run-length constraints; 5x5 test puzzle; 3 tests passing.

**exp1215: Milestone 2026.04.94 retro**
Uses the STEP 0 skeleton fix: write artifact skeleton as the FIRST action.
Routes to claude/opus (not codex). max_turns:100. Reads all exp1203-exp1214 results.

---

## Dependency Graph

```
exp1203 (pre-test fix)
  → ALL experiments that omit skip_pre_test:true depend on this
exp1204 (retro template) → exp1215 uses the documented STEP 0 pattern
exp1205 (paper issues) → exp1206 (bundle v8)
exp1207 (llama.cpp) → exp1208 (GRPO v5)
exp1209, exp1210, exp1211, exp1212, exp1213, exp1214 — unconditional
exp1215 (retro) — reads all previous
```

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|---|---|---|
| exp1203, 1204 | No | CPU diagnostics |
| exp1205, 1206 | No | LaTeX + file edits |
| exp1207 | Yes (CUDA) | llama.cpp GPU offload verification |
| exp1208 | Yes (2x RTX 3090) | DualGPU MANDATORY |
| exp1209, 1210, 1212, 1213, 1214, 1215 | No | CPU / llama.cpp CPU |
| exp1211 | Yes (GPU) | GGUF inference Qwen3.6-35B |

---

## Success Criteria (13 total)

1. `pre_test_suite_passing` (exp1203) — >= 400 tests pass after fix
2. `retro_template_updated` (exp1204) — STEP 0 pattern documented
3. `critical_issues_fixed_5_of_5` (exp1205) — gates exp1206
4. `arxiv_bundle_v8_ready` (exp1206) — PDF compiled, bundle packaged
5. `llama_cpp_gpu_offload_v3_verified` (exp1207) — >= 50 tok/s
6. `grpo_v5_honest_result` (exp1208) — any honest result (including blocked)
7. `grpo_vps_step_delta_measured` (exp1209) — delta measured (any sign)
8. `phase4_bfs_intractable_fraction_above_50pct` (exp1210)
9. `fover_v7_pairs_above_500` (exp1211)
10. `tier1_online_addition_honest_verdict` (exp1212)
11. `sdpo_dense_reward_delta_measured` (exp1213)
12. `nonogram_cartridge_shipped` (exp1214)
13. `retro_complete` (exp1215)

---

## Key Architectural Decisions for .94

1. **skip_pre_test: true on all .93-MISSING retries** until exp1203 confirms the test suite
   is working. Prevents 3x SKIP → RETIRE cascade before the root cause is fixed.

2. **STEP 0 skeleton in every opus/heavy task**: write `status="in_progress"` artifact JSON
   FIRST, then do work. Ensures deliverable exists even if agent times out.

3. **Retro: claude/opus, max_turns:100, STEP 0**: reverts AGENT_TYPE_RETRO=codex that
   caused the last two retro bootstrap failures.

4. **All carry-forwards pre-populate prior_failures**: classifies scope overlaps at plan time.
   Zero operator-intervention DOOMED_RERUN_BLOCK recoveries expected in .94.

5. **No gemini agent_type** (429-rate-limited per known-issues.md).

6. **GRPO v5 still gated on GPU offload**: prevents 60+ min wasted wall time.

7. **Phase 5 derisking (committed 2026-05-03)**: exp1208 (GRPO v5) + exp1213 (SDPO) are
   experiments 1 and 2 of the 4-experiment in-situ training validation sequence.

---

## New Research References Filed (2026-05-03)

| Paper | arxiv ID | Filed for |
|---|---|---|
| SDPO Self-Distilled Policy Optimization | 2604.03128 | exp1213 in .94 |
| ReProbe Internal State Probing | 2511.06209 | .94/.95 verifier comparison |
| CANUF Differentiable Constraint Layer | 2601.12442 | .95+ GRPO constraint training |
| BRAIN Boltzmann RL for Ising | 2602.09162 | .96+ hardware noise robustness |
| Spurious Rewards ICLR 2026 | 2512.16912 | .95 GRPO v6 design context |
