# Carnot — Session Metrics

## Session: 2026-04-20 Milestone 2026.04.45 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 2 | 2026-04-20T16:41:02Z | 2026-04-20T16:48:59Z | Exp 598: HISR + D-Wave integration — created hisr_weights.py, dwave_backend.py, experiment_598_hisr_dwave.py, tests; added REQ-LEARN-072 + REQ-SAMPLE-034 to specs; 25/25 tests pass, 100% coverage on new files | ~15k |
| 1 | 2026-04-20T13:45:00Z | 2026-04-20T14:45:21Z | Planned milestone 2026.04.45 "Live-Calibrated CoACE and DSVD — Closing the Offline/Live Distribution Gap": read 11 project files + .44 retro JSON (Exp 588) + Exps 577-587 results. Spawned arxiv research subagent (5 new papers: arXiv 2603.01025 OTV one-token LoRA verifier; arXiv 2604.02341 PROGRS outcome-conditioned PRM centering; arXiv 2604.10693 FACT-E factuality EBM; arXiv 2604.01564 p-bit synchronous Ising; arXiv 2604.05042 EBM dynamical models tutorial). Appended 5 new entries to research-references.md as "2026-04-20 arxiv Scan (Milestone 2026.04.45 Planning)". Identified 3 biggest gaps: (1) RETRO-066 CRITICAL — CoACE offline/live distribution gap: 86.7% offline vs 5.9% live recall; (2) JEPA v11 AUC=1.0 on only 9 synthetic pairs, likely overfitting, needs live corpus validation; (3) RETRO-033 #13 still unresolved — live verify-repair never showed positive improvement. Wrote openspec/change-proposals/research-roadmap-v45.md (12 experiments 589-600, 5 phases: Infrastructure/Live-Calibrated Extraction/JEPA v12 Retrain/Live Verify-Repair/Self-Learning+Research; dependency graph; success criteria; DSVD path as RETRO-033 alternate; OTV verifier integration). Wrote research-roadmap-next.yaml (12 experiments with full prompts; Exp 591 gates 594; Exp 592 gates 595; Exp 594 OR 595 gates 596; ExclusionManifest wire-in in Exp 589). | ~90k |

---

## Session: 2026-04-20 Milestone 2026.04.44 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-20T09:15:00Z | 2026-04-20T10:59:16Z | Planned milestone 2026.04.44: read 11 project files + .43 retro JSON (results/experiment_574_retro_2026_04_43.json) + Exps 569/567/568 results. Spawned arxiv research subagent (12 candidates, 6 genuinely new: arXiv 2604.10660 CPMI hard-negative contrastive pairs — DIRECT fix for RETRO-063 JEPA anti-correlation; arXiv 2503.03149 DSVD mid-generation rollback — new Tier 2.5; arXiv 2604.11611 MISE hindsight MI self-eval — filed .45; arXiv 2603.18683 HISR segment-level rewards — filed .45; arXiv 2602.13551 FLIP backward inference rewards — filed .45; arXiv 2601.22642 Interleaved Formal-Logic Verification — filed .45). Appended 6 new entries to research-references.md. Identified 3 biggest gaps: (1) RETRO-064 CRITICAL — CoACE v1 recall=5.9% (3/51 TP) because IT models write multi-step chains + prose arithmetic; fix: CoACE v2 with multi-step chain tracking, prose patterns, numeric context dict; (2) RETRO-063 CRITICAL — JEPA anti-correlated (AUC≤0.4444) across 3 consecutive retrains even after PURE min-form fix; fix: CPMI arXiv 2604.10660 explicit contrastive pairs via hard-negative mining; (3) RETRO-062 third miss — Live 50q A never collected; fix: import-time assertion before transformers/torch imports. Wrote openspec/change-proposals/research-roadmap-v44.md (14 experiments 575-588, 7 phases: Phase 0 Exclusion Manifest, Phase 1 Recall Surgery, Phase 2 Live Data, Phase 3 JEPA Retrain, Phase 4 Integration, Phase 5 FPGA, Phase 6 New Research, Phase 7 Retro; dependency graph; success criteria table; hardware requirements; DSVD Tier 2.5 in cascade diagram). Wrote research-roadmap-next.yaml (14 experiments: Exps 575-588 with complete prompts; gating: Exp 581 gates Exps 582+583; Exp 580 JEPA v11 CPMI retrain; Exp 584 KV260 Vivado synthesis gates Exp 585). | ~95k |

---

## Session: 2026-04-20 Milestone 2026.04.43 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-20T07:25:51Z | 2026-04-20T07:41:38Z | Planned milestone 2026.04.43: read 11 project files + .42 retro JSON; arxiv research subagent (6 new papers: 2510.04081 Caco CoACE, 2504.15275 PURE min-form PRM, 2509.10753 HalluField, 2604.09482 PRA EBM beam search, 2602.18145 freq-attention, 2603.23854 Symbolic-KAN); identified 3 biggest gaps (RETRO-061 extraction TP=0, RETRO-060 JEPA AUC=0.4286 anti-correlated, RETRO-062 Live 50q A unrun); wrote research-roadmap-v43.md (12 exps 563-574, 7 phases, dependency graph, CoACEExtractor + PURE loss + HalluField Tier 0e + PRA beam search); wrote research-roadmap-next.yaml (12 complete prompts); updated research-references.md | ~90k |

---

## Session: 2026-04-20 Milestone 2026.04.42 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-20T02:32:58Z | 2026-04-20T02:48:56Z | Planned milestone 2026.04.42: read 11 project files + .41 retro JSON + Exp 538 result; arxiv research subagent (1 new paper: 2603.20224); identified 3 biggest gaps (RETRO-058 synthetic epidemic, RETRO-033 no-improvement root cause, RETRO-056 JEPA anti-correlated); wrote research-roadmap-v42.md (14 experiments 549-562, 6 phases); wrote research-roadmap-next.yaml (14 complete prompts); updated research-references.md, changelog.md, status.md | ~85k |

---

## Session: 2026-04-16 Exp 413 EnvironmentAutoFix + RETRO-022

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T12:59:06Z | 2026-04-16T13:10:27Z | Exp 413: implemented EnvironmentAutoFix (apply_env_autofix, build_env_autofix_artifact); REQ-INFRA-021/022 + SCENARIO-INFRA-025/026/027 added to spec; exported from carnot.pipeline.__init__; experiment_413_env_autofix.py; 38 tests pass (100% targeted coverage); results/experiment_413_env_autofix.json written — honest_verdict=auto_fix_applied, retro_022_resolved=True; RETRO-022 CLOSED (workaround); conductor-log.md updated | ~28k |
| 2 | 2026-04-16T17:16:45Z | 2026-04-16T17:36:37Z | Exp 425: ExperimentTimeoutWatchdog — implement RETRO-003 45-min hard cap; REQ-INFRA-023/024 + SCENARIO-INFRA-028/029/030; experiment_watchdog.py; 35 tests pass (100% targeted coverage); experiment_425 script; ops docs; RETRO-003 CLOSED | ~20m |
| 3 | 2026-04-16T21:42:37Z | 2026-04-16T22:24:54Z | Exp 427: confirm/re-run Exp 419 precision benchmark harness; Exp 419 status='partial' → rerun path; compute_crane_detection_rate(); build_exp427_artifact(); check_dual_gpu_health() zombie/thermal gates; ExperimentTimeoutWatchdog(90min); 35 tests pass; 2541 total pass; 2 pre-existing failures unchanged; LIVE RUN PENDING | ~42m |
| 4 | 2026-04-16T22:33:05Z | 2026-04-16T22:53:37Z | Exp 428: confirm/re-run Exp 420 HumanEval live benchmark; Exp 420 status='partial' → full re-run harness; apply_env_autofix() module-level (RETRO-022 mitigation); 4-gate sequence (Gate0 informational, Gate1 LiveGPUGate, Gate2 dual-GPU warning, Gate3 setup_gpu, Gate4 model load); ExperimentTimeoutWatchdog(60min); Exp 226 baseline target embedded; gate+autofix metadata in artifact; 24 new tests pass (100% new-function coverage); 184 related tests all pass; LIVE RUN PENDING | ~20m |
| 5 | 2026-04-16T23:23:32Z | 2026-04-16T23:41:51Z | Exp 429: adversarial GSM8K live benchmark (Apple arXiv 2410.05229); Exp 421 status='partial' → full re-run harness; apply_env_autofix() module-level; Gate 0-4 chain (same as Exp 428); ExperimentTimeoutWatchdog(75min); _run_three_conditions() (standard/adversarial/repaired, 50q, checkpoint/10); adversarial_drop_pct + repair_improvement_pct headline metrics; schema='carnot.adversarial_gsm8k.v2'; reuses helpers from Exp 355+368; 42 new tests pass; LIVE RUN PENDING | ~18m |
| 6 | 2026-04-17T03:45:59Z | 2026-04-17T04:02:49Z | Exp 432: JitRL live validation; read existing scripts+tests (already scaffolded); ran experiment (synthetic_fallback, fp_reduction=33.71%); 39 tests pass; REQ-LEARN-034/SCENARIO-LEARN-060/061 confirmed; ops docs updated (conductor-log, changelog, status, metrics); Tier 1 self-learning partially met | ~17m |
| 7 | 2026-04-17T05:53:12Z | 2026-04-17T05:54:17Z | Exp 435 doc update: removed duplicate changelog entry (actual commit already had full details); updated status.md row (NPUPrereqResult, IRON toolchain details, REQ-PRED-005 new); appended traceability.md rows for REQ-PRED-005 (IRON sampler) + SCENARIO-EXP303-G (IRON path success); metrics updated | ~2m |
| 8 | 2026-04-17T06:00:57Z | 2026-04-17T06:05:18Z | Exp 435a: Phase 3 seed — continuous EBM vs Ising; REQ-KONA-001 + SCENARIO-KONA-001/002 added to spec; python/carnot/phase3/continuous_ebm.py; 33 tests pass; experiment script runs in 0.2s; honest_verdict=partial_match (sign_agreement=0.8, L2=2.69) | ~5m |
| 9 | 2026-04-17T09:24:44Z | 2026-04-17T09:46:00Z | Exp 439: live precision micro-benchmark harness; precision_micro.py (MicroPrecisionResult + build_micro_precision_artifact, 100% coverage); experiment_439_live_precision_micro.py (3 variants × 2 models × 50q; CRANE_ONLY + FULL_STACK with JitRL; CoT log); 33 tests all pass; REQ-BENCH-009 + SCENARIO-BENCH-025/026 added to spec; ops docs updated | ~21m |
| 10 | 2026-04-17T10:15:13Z | 2026-04-17T10:24:46Z | Exp 439 verification: read all existing code; confirmed harness fully implemented from prior session; 33 tests re-run, all pass; no changes required | ~10m |
| 11 | 2026-04-17T11:55:37Z | 2026-04-17T12:35:51Z | Exp 440: read existing humaneval_micro.py + experiment_440 script + test file; 46 tests pass; traceability updated (REQ-BENCH-009/010, SCENARIO-BENCH-025-028); changelog + status updated | ~40m |
| 12 | 2026-04-17T13:35:58Z | 2026-04-17T13:47:41Z | Exp 441 adversarial micro-benchmark: verified all code already exists (adversarial_gsm8k.py MicroAdversarialResult/build_micro_adversarial_artifact, experiment_441 script, 40 tests); REQ-BENCH-011/SCENARIO-BENCH-029/030 confirmed in spec; ran full test suite (3926 passed, 2 pre-existing failures unrelated to Exp 441); LIVE RUN PENDING | ~12m |
| 13 | 2026-04-17T15:16:27Z | 2026-04-17T15:28:43Z | Exp 442: FOVER live CoT annotation — read fover_annotator.py, env_autofix.py, Exp 439 live data; verified fover_live.py + experiment_442 script already exist; ran 63 tests (all pass); executed experiment (300 live responses → 57 labeled pairs; honest_verdict=real_data_labeled); updated conductor-log.md + changelog.md; FR-11 upstream relay condition met for first time after 8 milestones of synthetic_only | ~12m |
| 14 | 2026-04-17T19:09:10Z | 2026-04-17T19:23:59Z | Exp 444: CarnotThinkProbe — verified all code already implemented (think_probe.py, test_think_probe.py, experiment_444 script, __init__ exports, verify_repair integration, spec REQ-VERIFY-094/095, architecture Tier 0 table); 56 tests pass; 99 combined tests (think+sink probe) pass; changelog/metrics updated | ~14m |
| 15 | 2026-04-17T20:14:25Z | 2026-04-17T20:37:53Z | Exp 445: BoltzmannRepairBridge — new module boltzmann_repair.py (RepairDirection, LinearSpinAdapter, BoltzmannRepairBridge); 30 tests pass (100% targeted coverage); experiment_445 script; exported from __init__; REQ-REPAIR-014/015 + SCENARIO-REPAIR-028/029/030 added to spec + traceability; 3359 existing tests pass, 2 pre-existing failures unrelated | ~23m |
| 16 | 2026-04-18T00:40:11Z | 2026-04-18T01:03:14Z | Exp 446: Langevin dynamics + Energy Matching — verified prior conductor implementation; read continuous_ebm.py (sample_langevin, sample_energy_matching, compare_samplers), test_experiment_446_energy_matching.py (36 tests), experiment_446 script, spec (REQ-KONA-002/003 + SCENARIO-KONA-003/004/005), architecture.md, conductor-log.md, changelog.md — all already updated; ran 36 tests: all pass; 8746 total tests collected in suite; CPU-only, Phase 3 seed | ~23m |
| 17 | 2026-04-18T03:41:34Z | 2026-04-18T03:49:11Z | Exp 447: KAEMEnergy exact inverse-transform sampling — verified prior conductor implementation; read kaem_energy.py (UnivariateKAEMLayer, KAEMEnergy, benchmark_kaem_vs_mcmc), test_kaem_energy.py (51 tests), experiment_447 script, spec (REQ-SAMPLE-015/016 + SCENARIO-SAMPLE-027/028/029), models/__init__.py — all already complete; ran 51 tests: all pass, 100% coverage on kaem_energy.py; updated architecture.md (KAN fast-path tier section), conductor-log.md; CPU-only; arXiv 2506.14167 KAEM | ~8m |
| 18 | 2026-04-18T05:11:54Z | 2026-04-18T05:20:00Z | Exp 447 re-verification (human session): re-read all kaem_energy.py, tests, experiment script, spec, __init__.py, conductor-log; confirmed all already implemented; re-ran 51 tests: all pass; full suite running in background; no new code changes needed | ~8m |
| 19 | 2026-04-18T06:42:16Z | 2026-04-18T06:53:10Z | Exp 447 human-session verification: read all existing KAEM code (kaem_energy.py, test_kaem_energy.py, experiment_447, spec, conductor-log, changelog, architecture.md); all already fully implemented from prior sessions; 51 tests pass; ran experiment_447 script (honest_verdict=no_speedup, mean_speedup=1.29x — n_vars={10,25,50,100}, n_samples=100); confirmed REQ-SAMPLE-015/016 + SCENARIO-027/028/029 in spec; task IDs 011/012/023/024/025 in spec are pre-existing FPGA requirements (not KAEM); all ops docs already updated | ~11m |

---

## Session: 2026-04-19 Milestone 2026.04.41 Research Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-19T22:46:37Z | 2026-04-19T22:56:05Z | Research planning for milestone 2026.04.41: read 10 project files (research-program.md, prd.md, architecture.md, status.md, changelog.md, research-roadmap.yaml, research-complete.yaml partial, research-references.md, research-hardware-wishlist.md, conductor-log.md, v40 roadmap, retro JSON); spawned arxiv research agent (found 2 new papers: 2511.07124 EBM-CoT calibration, 2511.06209 internal-state probes); identified 3 biggest gaps (RETRO-055 live inference latency, ExperimentTemplate.teardown() unimplemented 5th time, self-learning still synthetic-only); wrote openspec/change-proposals/research-roadmap-v41.md (milestone 2026.04.41, 12 exps 537-548, 8 phases); wrote research-roadmap-next.yaml (12 experiments, all prompts complete); appended 2 new entries to research-references.md | ~45k |

---

## Session: 2026-04-19 Milestone 2026.04.36 Research Planning (v42)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-19T01:49:45Z | 2026-04-19T01:59:50Z | Research planning for milestone 2026.04.36: read 10 project files (research-program.md, prd.md, architecture.md, status.md, changelog.md, research-roadmap.yaml, research-hardware-wishlist.md, conductor-log.md, research-references.md partial) + operational retro 2026.04.35 JSON; 1 arxiv web search (10 topics, 2025-2026); identified 3 biggest gaps (zombie VRAM blocks live benchmarks 3rd consecutive milestone, JEPA regression 0.667→0.400, GPU 1 idle 100% of milestone); wrote openspec/change-proposals/research-roadmap-v36.md (milestone 2026.04.36, 13 exps 474-486, 5 phases); wrote research-roadmap-next.yaml (13 experiments, all prompts complete); appended 5 new entries to research-references.md | ~65k |

## Session: 2026-04-19 Milestone 2026.04.39 Research Planning (v44)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-19T14:00:00Z | 2026-04-19T15:35:57Z | Research planning for milestone 2026.04.39: read 10 required project files + operational retro 2026.04.38 JSON; arxiv subagent search (4 new papers added to research-references.md: Hallucination Basins 2604.04743, LeWorldModel JEPA 2603.19312, Constrained Decoding 2604.14862, Low-rank Energy Landscape 2604.04384); identified 3 biggest gaps (RETRO-051 JIT VRAM at model.load(), RETRO-052 DualGPU 0 scripts on cuda:1, FR-11 synthetic-only 9th milestone); wrote openspec/change-proposals/research-roadmap-v39.md ("Close the Credibility Gap — JIT VRAM, Seventh Attempt, DualGPU Verified", 8 phases, 12 exps 513-524, dependency graph, success criteria table); wrote research-roadmap-next.yaml (12 experiments, all prompts complete); updated ops/status.md + ops/changelog.md; did NOT modify research-roadmap.yaml or scripts/research_conductor.py | ~80k |

---

## Session: 2026-04-19 Milestone 2026.04.40 Research Planning (v45)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-19T19:26:22Z | 2026-04-19T19:26:35Z | Research planning for milestone 2026.04.40: read 10 required project files + operational retro 2026.04.39 JSON; arxiv scan (5 new papers added to research-references.md: Adaptive Rectification Sampling 2504.01317, Potts Machine 2602.04200, GRPO Verifiable Rewards 2503.06639, IR3 Contrastive IRL 2602.19416, AutoRefine 2601.22758); identified 3 biggest gaps (RETRO-053 env_autofix '0' not overridden — single gate blocking RETRO-033 eighth attempt, GPU1 at 0% forward-pass compute RETRO-052, validated NUP Probe v4 + Basin Detector not wired into cascade); wrote openspec/change-proposals/research-roadmap-v40.md ("Fix the Last Gate — Eighth Attempt, First Live Positive", 7 phases, 11 exps 526-536, updated architecture diagram with Tier 0c/0d, success criteria table, arxiv papers table); wrote research-roadmap-next.yaml (11 experiments, all prompts complete, {project_root}/{date} placeholders); did NOT modify research-roadmap.yaml or scripts/research_conductor.py | ~90k |

## Session: 2026-04-19 Milestone 2026.04.37 Research Planning (v43)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-19T05:30:15Z | 2026-04-19T05:45:03Z | Research planning for milestone 2026.04.37: read 10 required project files + operational retro 2026.04.36 JSON; arxiv subagent search (3 new papers: LLM-JEPA 2509.14252, Bayesian Semantic Entropy 2603.22812, SuRe 2511.22367); prepended new section to research-references.md; identified 3 biggest gaps (RETRO-044 GPUVRAMGate kill order, JEPA AUC regression 0.667→0.281 via majority-class collapse, GPU 1 at 11% utilization); wrote openspec/change-proposals/research-roadmap-v37.md ("Break the VRAM Deadlock", 6 phases, 13 exps 487-499); wrote research-roadmap-next.yaml (13 experiments, all prompts complete); did NOT modify research-roadmap.yaml or scripts/research_conductor.py | ~75k |
| 2 | 2026-04-19T06:11:43Z | 2026-04-19T06:13:10Z | Minimal doc updates for Exp 488 (Live 100q Precision v5 — RETRO-033 fifth attempt with GPUVRAMGateV2): appended changelog entry, appended status table row, appended 6 new REQ-/SCENARIO- rows to traceability.md (REQ-BENCH-034/035/036, SCENARIO-BENCH-053/054/055); spec updated with new benchmarking requirements; no modifications to research conductor or roadmap | ~2m |

---

## Session: 2026-04-18 Milestone 2026.04.35 Research Planning (v41)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-18T21:26:42Z | 2026-04-18T21:43:08Z | Research planning for milestone 2026.04.35: read 11 project files + operational retro 2026.04.34; 6 arxiv/web searches (PPSEBM, GPU-OIM, EP-OIM, GSM-Symbolic, sparsified FPGA Ising, self-adaptive Ising); appended 6 new entries to research-references.md; identified 3 biggest gaps (process debt, live benchmark scale, self-learning AUC plateau); wrote openspec/change-proposals/research-roadmap-vNEXT.md v41 (milestone 2026.04.35, "Scale the First Positive", 12 exps 462-473, 3 phases); wrote research-roadmap-next.yaml (951 lines, 12 experiments, all prompts complete) | ~60k |

---

## Session: 2026-04-18 Milestone 2026.04.34 Research Planning (v40)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-18T11:30:11Z | 2026-04-18T11:34:41Z | Research planning for milestone 2026.04.34: read 11 project files; 8 arxiv/web searches (VeriCoT, VPRM, LSEBMCL, EBM-CoT, KAEM, AMD XDNA IRON, Gemma4 bug); appended 5 new entries to research-references.md; wrote openspec/change-proposals/research-roadmap-vNEXT.md v40 (milestone 2026.04.34, 12 exps 450-461, 5 phases); wrote research-roadmap-next.yaml (812 lines, 12 experiments, all prompts complete) | ~35k |
| 2 | 2026-04-18T14:19:40Z | 2026-04-18T14:47:25Z | Exp 452: RETRO-030 closure — AtomicResultWriter + Energy Matching v2. Read env_autofix.py, experiment_template.py, experiment_446 script, known-issues.md, __init__.py, spec.md. Added REQ-INFRA-031/032 + SCENARIO-INFRA-039/040 to spec. Wrote tests/python/test_atomic_writer.py (11 tests). Implemented python/carnot/pipeline/atomic_writer.py (AtomicResultWriter: write() = json.dumps → .tmp → os.rename, verify_exists()). Exported from carnot.pipeline.__init__. Wrote scripts/experiment_452_energy_matching_v2.py (re-runs Exp 446 logic, AtomicResultWriter, verify_exists() guard, Phase 3 improvement tracking). 11 tests pass, 100% targeted coverage. Updated conductor-log.md, changelog.md, status.md, known-issues.md. RETRO-030 CLOSED. | ~28m |

---

## Session: 2026-04-16 Milestone 2026.04.31 Research Planning (v37)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T12:44:07Z | 2026-04-16T12:56:17Z | Read 10 project files (research-program.md, prd.md, architecture.md, status.md, changelog.md, research-complete.yaml, research-roadmap.yaml, conductor-log.md, research-references.md, hardware-wishlist.md); 8 arxiv web searches (EBMs, KANs, Ising machines, process reward models, VPRM, AMD NPU, thermodynamic computing, continual self-learning); appended 6 new papers to research-references.md (VPRM 2601.17223, FOVER 2505.15960, ThinkPRM 2504.16828, AMD NPU IRON 2504.03083, Thermodynamic Init 2603.24183, Self-Certainty BoN 2502.18581); wrote openspec/change-proposals/research-roadmap-vNEXT.md v37 (milestone 2026.04.31, 12 exps 413-424); wrote research-roadmap-next.yaml (12 experiments, 4 phases) | ~35k |

---

## Session: 2026-04-16 Exp 411 Live HumanEval Code Verification

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T11:52:29Z | 2026-04-16T12:20:37Z | Exp 411: read Exp 404 preflight (honest_verdict=env_not_propagating, not gpu_confirmed_live); implemented experiment_411_humaneval_live.py (Gate 0 preflight check, _load_preflight, _utc_now/_utc_date, _write_artifact, main with 4-gate sequence importing all HumanEval helpers from Exp 369); wrote 44 tests; full suite 3058 pass, 2 pre-existing failures; produced blocked artifact at results/experiment_411_humaneval_live.json; updated changelog + conductor-log + metrics | ~22k |

---

## Session: 2026-04-16 Exp 410 Live Precision Pipeline Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T10:46:18Z | 2026-04-16T11:01:02Z | Exp 410: read Exp 404 preflight (honest_verdict=env_not_propagating, not gpu_confirmed_live); implemented experiment_410_precision_live.py (load_preflight_verdict, build_exp410_artifact, _write_artifact, main with 4-gate sequence + CRANE→LLM fallback); wrote 34 tests covering all new functions; ran 56 tests pass; produced blocked artifact; updated changelog + metrics | ~18k |

---

## Session: 2026-04-16 Milestone 2026.04.30 Research Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T05:00:00Z | 2026-04-16T05:59:00Z | (prior context) Read 11 project files; arxiv research background agent; appended 7 new papers to research-references.md; wrote openspec/change-proposals/research-roadmap-vNEXT.md v36 (395 lines, milestone 2026.04.30 design); wrote research-roadmap-next.yaml (649 lines, 14 experiments Exp 404-417) | ~80k |
| 2 | 2026-04-16T10:00:49Z | 2026-04-16T10:01:02Z | Context compaction resume: verified both output files exist and contain correct content; logged metrics | ~2k |

---

## Session: 2026-04-16 Exp 389 Milestone 2026.04.28 Retrospective

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T06:45:14Z | 2026-04-16T07:02:16Z | Exp 389: wrote experiment_389_retro_2026_04_28.py (MilestoneRetro2026_04_28 dataclass, compute_retro_2026_04_28, build_retro_artifact v3, estimate_speedup_pct, load_milestone_results, compute_timing_stats, _check_cikan_implemented, main); wrote test_experiment_389_retro.py (115 tests); ran script to produce results/operational_retro_2026_04_28.json; updated ops/status.md + ops/changelog.md + ops/conductor-log.md; 4 pre-existing failures in full suite (unrelated: tests 319, 337, 295) | ~22k |

---

## Session: 2026-04-16 Exp 380 Live HumanEval Execute

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T04:41:21Z | 2026-04-16T04:55:00Z | Exp 380: wrote experiment_380_humaneval_execute.py (uses LiveGPUGate.require_live_or_blocked + imports all helpers from Exp 369, no duplication); wrote 24 tests covering all 4 gate paths; 115 pass (Exp 379+380+369 combined); updated conductor-log.md + metrics.md | ~14k |

---

## Session: 2026-04-16 Exp 379 Live Precision Pipeline Execution

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T04:10:01Z | 2026-04-16T04:35:24Z | Exp 379: wrote experiment_379_precision_execute.py (thin wrapper over Exp 368 pipeline: LiveGPUGate gate, import run_variant/load_gsm8k_questions, build_exp379_artifact with v2 schema + honest_verdict); wrote 22 tests (100% coverage of new functions); ran full suite (3204 pass, 2 pre-existing failures unrelated); updated conductor-log.md + metrics.md | ~12k |

---

## Session: 2026-04-15 Exp 367 Live Extraction Comparison Verification

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T22:44:31Z | 2026-04-15T23:03:55Z | Exp 367: Verified all implementation files exist (extractor_comparison.py, experiment_367_extraction_live.py, test_experiment_367_extraction_live.py). REQ-EXTRACT-023 + SCENARIO-EXTRACT-047/048 confirmed in spec. Ran full test suite: 6577 passed, 80 pre-existing failures in test_experiment_319_retro.py (unrelated). Exp 367 + Exp 358 tests: 75 passed. Updated ops. | ~4k |
| 2 | 2026-04-15T23:07:03Z | 2026-04-15T23:17:29Z | User-requested re-run: read all Exp 367 implementation files; confirmed all already complete; ran test_experiment_367_extraction_live.py (42 pass) + full suite (2912 pass, 1 pre-existing failure in test_experiment_319_retro.py unrelated to 367); reconciled spec (REQ-EXTRACT-023/SCENARIO-EXTRACT-047/048 confirmed present); honest_verdict=live_gpu_winner gated on ALL results live_gpu (stricter than Exp 358). | ~6k |

---

## Session: 2026-04-15 Exp 365 RETRO Close

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T21:58:43Z | TBD | Exp 365: Close RETRO-012/013/014 — build ConductorEnvFix + conductor_gpu_env.sh, RetroJSONEnforcer, RetroItemTracker; spec REQ-INFRA-015/016 + SCENARIO-INFRA-016/017/018; 100% test coverage | TBD |

---

## Session: 2026-04-16 Milestone 2026.04.28 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T03:03:23Z | 2026-04-16T03:19:11Z | Plan next milestone 2026.04.28: read 11 project files, arxiv research (6 new papers: Physical Analog KAN 2602.07518, BiKA 2602.23455, JitRL 2601.18510, Ising↔NN 2511.00746, Adaptive Rejection Sampling 2504.05410, REGREACT 2604.12054), identified 3 gaps (RETRO-015 live GPU 4th milestone, RETRO-018 CIKAN corrupt, JitRL reveals correct Tier 1 algorithm), designed 13 experiments (377-389) across 5 phases, wrote research-roadmap-vNEXT.md (v34) + research-roadmap-next.yaml | ~10k |

---

## Session: 2026-04-16 Milestone 2026.04.29 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-16T07:00:00Z | 2026-04-16T07:42:31Z | Plan next milestone 2026.04.29: read 11 project files, arxiv research (5 new papers: Semantic Energy 2508.14496, CRANE 2502.09061, DSP/Feasibility Channels 2604.02350, Potts MFC 2602.04200, LLM-QUBO 2509.00099), added papers to research-references.md, identified 3 gaps (RETRO-019 GPU offline 5th milestone, RETRO-020 CIKAN+JitRL+SafetyKAN missing, RETRO-021 FR-11 unconfirmed), designed 14 experiments (390-403) across 5 phases + mandatory GPU preflight gate, wrote openspec/change-proposals/research-roadmap-vNEXT.md (v35) + research-roadmap-next.yaml | ~18k |

---

## Session: 2026-04-15 Milestone 2026.04.27 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T21:47:22Z | 2026-04-15T21:56:02Z | Plan next milestone 2026.04.27: read 10 project files, arxiv research (4 new papers added to research-references.md: CIKAN 2412.03710, Thermodynamic init 2603.24183, RLVR 2506.14245, StructEval 2505.20139), identified 3 gaps (RETRO-012/live GPU, LLMExtractor, real-data self-learning), designed 12 experiments (365-376) across 5 phases, wrote research-roadmap-vNEXT.md (v33) + research-roadmap-next.yaml | ~9m |

---

## Session: 2026-04-15 Exp 359 EORM Real-Data Retrain

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T18:23:15Z | 2026-04-15T18:44:36Z | Exp 359: verified all implementation files exist; ran 48 tests (pass, 100% eorm_retrain.py coverage); fixed _pairs_to_contrastive_triples bug (synthetic_* IDs now routed to shared pool); ran full experiment: 50 epochs, 60 triples, before_auc=0.500, after_auc=0.500, honest_verdict=synthetic_only; saved model; updated spec traceability, conductor-log, changelog, status.md | ~21m |
| 2 | 2026-04-15T19:27:00Z | 2026-04-15T19:40:26Z | Exp 361: three-tier self-learning relay (FR-11 mandatory). Read 6 existing files; added REQ-LEARN-026/027 + SCENARIO-LEARN-045/046/047 to spec; created self_learning_relay.py (SelfLearningBatchResult, SelfLearningRelay, _compute_auc_roc, compute_learning_improvement, build_relay_artifact); 54 tests pass, 100% new-module coverage; experiment_361 ran: batch1=0.600→batch4=0.720, improved=True, all 4 Tier 2 templates activated; honest_verdict=synthetic_only; updated conductor-log, changelog, status.md, traceability | ~13m |

---

## Session: 2026-04-15 Exp 355 Adversarial GSM8K Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T16:49:59Z | 2026-04-15T17:01:43Z | Exp 355: adversarial GSM8K benchmark — SCENARIO-BENCH-017/018/019 added to spec; run_adversarial_benchmark + _compute_top_level_verdict + main() written in experiment_355; 51 tests pass (100% targeted coverage); ops/conductor-log/changelog/status updated | ~12m |

---

## Session: 2026-04-15 Exp 352 Live GPU Diagnostic Root-Cause Fix

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T15:45:25Z | 2026-04-15T15:48:16Z | Exp 352: Diagnose + fix silent simulated fallback; verified all files exist; 37 tests pass, 100% module coverage; updated conductor-log/changelog | ~3m |
| 2 | 2026-04-15T15:49:35Z | 2026-04-15T15:50:05Z | Verify Exp 352 docs complete — changelog (2 entries), status.md row, traceability.md REQ-INFRA-014/SCENARIO-INFRA-014/015 all present; no appends needed | ~1m |
| 3 | 2026-04-15T15:51:37Z | 2026-04-15T15:57:00Z | Exp 352: Full end-to-end run — read all key files, ran tests (37 pass 100% module coverage), ran experiment_352 script; ROOT CAUSE CONFIRMED: carnot_force_live_set=False (CARNOT_FORCE_LIVE=1 not in conductor subprocess env); GPU hardware fully capable (cuda_visible=True, torch_available=True, model_loadable=True); updated conductor-log + changelog with diagnostic findings | ~6m |

---

## Session: 2026-04-15 Research Planning Milestone 2026.04.26

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T15:10:04Z | 2026-04-15T15:20:06Z | Plan next milestone 2026.04.26 — arxiv scan + design; 13 experiments (Exps 351-363) across 5 phases; 4 new papers added to research-references.md (ARM-EBM bijection 2512.15605, SAVeR 2604.08401, MathAgent 2604.11188, T-SKM-Net 2512.10461); 3 gaps: live GPU never works, constraint extraction broken for IT models, Apple adversarial GSM8K untested | ~10m |

---

## Session: 2026-04-15 Exp 347 JEPA Real-Data Retrain

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T13:31:45Z | 2026-04-15T13:48:36Z | Exp 347: JEPA real-data retrain on live violation pairs; REQ-LEARN-024 + SCENARIO-LEARN-041/042 to spec; ViolationPair + extract_violation_pairs + JEPARetrainer + build_retrain_artifact in jepa_retrain.py; 48 tests; experiment_347_jepa_real_retrain.py; ops docs updated | ~17m |

---

## Session: 2026-04-15 Exp 346 EORM CoT Energy Reward Model

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T12:32:27Z | 2026-04-15T13:28:29Z | Exp 346: EORM CoT energy reward model; REQ-LEARN-022/023 + SCENARIO-LEARN-038/039/040 to spec; EORMModel + EORMTrainer + CoTEnergyInput in eorm.py (pure JAX transformer, hash tokenizer, safetensors save/load); experiment_346_eorm_training.py; 52 tests 100% eorm.py coverage; ops docs + traceability updated | ~56m |

---

## Session: 2026-04-15 Exp 343 ConstraintTemplateLibrary

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T10:32:30Z | 2026-04-15T10:59:13Z | Exp 343: ConstraintTemplateLibrary — Tier 2 constraint addition from memory patterns; REQ-LEARN-017/018 + SCENARIO-LEARN-029/030/031/032 to spec; ConstraintTemplate dataclass + ConstraintTemplateLibrary (6 methods); 4 builtin template functions (carry_check, sign_check, unit_consistency, comparison_direction); wired into VerifyRepairPipeline; 66 tests all pass; experiment_343_constraint_templates.py + artifact written; ops docs updated | ~27m |

---

## Session: 2026-04-15 Exp 341 Live HumanEval Code Verification Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T09:18:33Z | 2026-04-15T09:46:51Z | Exp 341: Live HumanEval code verification benchmark; REQ-BENCH-004 + SCENARIO-BENCH-010/011 to spec; HumanEvalResult dataclass + compute_pass_at_1 + compute_pass_at_1_after_repair + build_humaneval_artifact; experiment_341_live_humaneval.py (50 problems, CI-safe simulated mode, CodeExtractor+VerifyRepairPipeline); 49 tests all pass; fixed 4 test-case bugs (HumanEval/0 args format, HumanEval/1 string tuple, HumanEval/20 unsorted list, HumanEval/36 wrong expected value); ops/changelog.md + ops/status.md updated | ~28m |

---

## Session: 2026-04-15 Exp 338 Host Prereqs + DualGPU Auto-Assignment

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T07:16:43Z | 2026-04-15T07:43:28Z | Exp 338: HostPrereqRegistry (RETRO-006) + DualGPU auto-assignment (RETRO-004); REQ-INFRA-006/007 + SCENARIO-INFRA-009/010/011 to spec; ops/host-prereqs.md (6 entries); host_prereq_registry.py; setup_gpu() dual_gpu_auto_assigned key; 75 tests all pass; results/experiment_338_host_prereqs.json; ops docs updated | ~27m |

---

## Session: 2026-04-15 Milestone 2026.04.25 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T07:02:07Z | 2026-04-15T07:13:47Z | Plan next milestone 2026.04.25 — arxiv scan (28 papers found, 6 new added to references); 3 gaps identified: new precision components not live-benchmarked, self-learning adds no new constraints, EORM/SinkProbe not yet integrated; 13 experiments (Exps 338-350) across 5 phases; research-roadmap-vNEXT.md (v31) + research-roadmap-next.yaml created | ~11m |

---

## Session: 2026-04-15 Exp 336 CoT Circuit Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T06:10:50Z | 2026-04-15T06:18:14Z | Exp 336: CoTCircuitVerifier (CRV arXiv 2510.09312); REQ-EXTRACT-015/016 + SCENARIO-EXTRACT-031–035 to spec; cot_circuit_verifier.py; 51 tests 100% module coverage; verify_cot_circuit() additive integration; experiment_336 benchmark script; all docs reconciled | ~7m |

---

## Session: 2026-04-15 Exp 334 VERGE-Style Iterative Z3 Refinement

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T05:32:03Z | 2026-04-15T05:39:16Z | Exp 334: VERGE iterative Z3 refinement; REQ-REPAIR-012/013 + SCENARIO-REPAIR-024–027 to spec; verge_refiner.py; 30 tests 100% coverage; verify_repair_verge(); experiment_334 script; traceability+changelog+status updated | ~7m |
| 2 | 2026-04-15T05:40:45Z | 2026-04-15T05:57:07Z | Verification turn: confirmed 30 verge_refiner tests pass, verge_refiner.py 100% coverage, VergeRefiner exported from pipeline __init__.py, all docs already reconciled by Turn 1 | ~16m |

---

## Session: 2026-04-15 Exp 332 Confidence-Weighted Repair

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T04:27:22Z | 2026-04-15T04:46:31Z | Exp 332: dual-signal confidence-weighted repair; REQ-VERIFY-083/084/085 + SCENARIO-109–112 to spec; 38 tests pass; confidence_weighted_repair.py; verify_repair_confidence_weighted(); Exp 332 benchmark FPs avoided 86.7%, TPs preserved 100%, GATE_EFFECTIVE | ~19m |

---

## Session: 2026-04-15 Exp 325 Conductor Hardening

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T01:28:11Z | 2026-04-15T01:53:08Z | Exp 325: conductor timeout wrapper (RETRO-001) + test-first stub (NEW-001); spec REQ-INFRA-001/002 + SCENARIO-INFRA-001/002/003; 23 tests pass; run_experiment_with_timeout.sh; generate_test_stub(); artifact all_checks_passed | ~25m |

---

## Session: 2026-04-15 Milestone 2026.04.24 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T01:12:49Z | 2026-04-15T01:25:30Z | Plan milestone 2026.04.24 — arxiv research (6 new papers), research-references.md updated, research-roadmap-vNEXT.md (v30) + research-roadmap-next.yaml created; 13 experiments (Exps 325-337) across 4 phases | ~12m41s |

---

## Session: 2026-04-15 Exp 319 Operational Retrospective

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T00:32:57Z | 2026-04-15T00:43:21Z | Exp 319: Operational retrospective for milestone 2026.04.23 — 59 tests, script, artifact; 4800 pass, 99.43% coverage | ~10m24s |

---


## Session: 2026-04-14 D-Wave Sampler Backend

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T22:59:00Z | 2026-04-14T23:18:07Z | D-Wave sampler: add [dwave] optional dep; create dwave_sampler.py (neal/tabu/qpu modes, BQM conversion, health_check, benchmark); register in get_backend factory; 41 tests, 74 sampler tests total pass | ~8k |

---

## Session: 2026-04-14 Conductor Audit Logging

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T22:29:32Z | 2026-04-14T22:36:11Z | Behavioral audit logging: read existing conductor_audit.py + test file; confirmed 52 tests pass; ran full suite: 4619 passed, 2 pre-existing failures (z3_gated_repair, experiment_template timeout), 99.45% coverage | ~5k |

---

## Session: 2026-04-14 Exp 318 Four-Tier Self-Learning Relay Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T21:02:57Z | 2026-04-14T21:15:49Z | Exp 318 four-tier relay: read Exps 302/309/312 code + results; add REQ-LEARN-013 + SCENARIO-LEARN-021/022 to spec; write 58 tests (TestConstants 2, TestRelayBatchResult 14, TestComputeRelayImprovement 6, TestSimulateGsm8kQuestions 6, TestRunRelayBatch 7, TestBuildRelayArtifact 20, TestConstraintDelta 3); implement experiment_318_self_learning_relay.py; run --simulated: B1=0.697 B2=0.545 B3=0.636 imp_1to3=-0.0606; 58 tests pass; update ops/changelog + status + traceability | ~15k |
| 2 | 2026-04-14T21:26:46Z | 2026-04-14T21:27:54Z | Minimal doc update: append Exp 318 row to ops/status.md (complete status, 4-tier relay detail); changelog/traceability already updated by commit | ~2k |

---

## Session: 2026-04-14 Exp 317 HuggingFace README Accuracy Audit

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T20:31:22Z | 2026-04-14T20:52:17Z | Exp 317 HF README audit: read Exp 304/293/316 code + results; add REQ-PUBLISH-003/SCENARIO-PUBLISH-005/006 to spec; write 46 tests (TestBuildPhase1ReadmePatch 7, TestPlaceholderCard 6, TestModelCardUpdateIdempotent 5, TestBuildFcvReadmeWithExp316 4, TestCredentialCheck317 4, TestBlockedArtifact317 6, TestRunExperiment317Schema 10, TestNoFakeUploads 2, TestPerTokenEbmRepoList 3, TestResultsJsonSchema317 7); implement experiment_317_hf_publish.py; 46 tests pass, 4390 total, 99.43% coverage; update ops/changelog + status | ~12k |
| 2 | 2026-04-14T20:59:34Z | 2026-04-14T21:00:45Z | Verification pass: confirmed experiment_317_hf_publish.py + test file complete; 46 pass, 7 skip (results file absent); REQ-PUBLISH-003/SCENARIO-PUBLISH-005/006 in spec; ops/changelog + status already updated | ~3k |

---

## Session: 2026-04-14 Exp 316 Full-Scale Benchmark Execution

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T20:23:53Z | 2026-04-14T20:28:50Z | Exp 316 full-scale benchmark execution: read Exp 315 script, write 28 tests (TestSchemaValidation 7, TestInferenceMode 2, TestCIBounds 2, TestSampleSize 3, TestPublishedBaselines 2, TestAccuracyRange 2, TestArtifactMetadata 5, TestLoadFullscaleResults 4); run benchmark --simulated 100 GSM8K + 20 HumanEval; 28 tests PASS; update ops/test-results + research-studying + ops/changelog + ops/status + _bmad/traceability | ~8k |

---

## Session: 2026-04-14 Exp 314 NPU Prereq Retry

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T19:53:23Z | 2026-04-14T20:07:19Z | Exp 314 NPU prereq retry: read Exp 303 script/results, hardware-wishlist; write 41 tests (TestExp314Schema 9, TestPrereqCheck314 6, TestPrereqChanges 5, TestBuildOutcome314 7, TestInferenceResult314 6, TestNoFabricatedLatency314 2); implement experiment_314_npu_prereq_install.py (_compute_prereq_changes, _attempt_source_build_314, _build_next_steps, _update_hardware_wishlist, main); run experiment→blocked_prereq; 4316 passed 99.45% coverage; update ops/changelog + status | ~16k |

---

## Session: 2026-04-14 Exp 313 KV260 Hardware Bring-Up

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T19:11:27Z | 2026-04-14T19:44:20Z | Exp 313 KV260 hardware bring-up: add REQ-SAMPLE-012+SCENARIO-SAMPLE-025/026 to training-inference/spec.md; write 40 tests (37 pass, 3 skip HW); implement experiment_313_kv260_bringup.py (detect_kv260_hardware, spin_validity_check, CPU fallback, AXI round-trip, honest_verdict); run experiment→blocked_no_bitfile; update ops/changelog, status, traceability, hardware-wishlist | ~18k |

---

## Session: 2026-04-14 PrefillUncertaintyProbe

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T10:12:57Z | 2026-04-14T10:20:44Z | Implement PrefillUncertaintyProbe (REQ-VERIFY-080): read spilled_energy_extractor, verify_repair, __init__; add spec entries SCENARIO-VERIFY-103/104; write 35 tests; implement prefill_uncertainty_probe.py (compute_input_uncertainty, compute_conjugate_bound, compute_prompt_uncertainty, PrefillUncertaintyProbe); add VerifyRepairPipeline.check_prefill_uncertainty(); export from __init__; 3644 total passed 99.12% coverage; update ops/changelog + status | ~12k |
| 2 | 2026-04-14T10:31:51Z | 2026-04-14T10:37:31Z | Exp 299 JEPA real logits retrain: read Exp 291 script, Exp 291 results, semantic_energy_extractor, existing tests; check for 294/295 logit files (absent); write 51 tests covering real logit loading/fallback, training_source, semantic_energy feature, isotonic calibration, conformal α=0.1, ONNX export/loadability, comparison_vs_exp291; implement experiment_299_jepa_real_logits.py; 51 passed; update ops/changelog + status | ~15k |
| 3 | 2026-04-14T11:15:28Z | 2026-04-14T11:27:09Z | Exp 302 integrated self-learning benchmark: read constraint_generator.py, confidence_verifier.py, case_memory.py, verify_repair.py, experiment_235 pattern; write 62 tests (PerQuestionRecord, BatchResult 50-question enforcement, ConstraintGenerationSummary, compute_improvement_delta honest negatives, count_dynamic_constraints, simulate_gsm8k_questions, run_batch, run_constraint_generation, build_artifact full schema); implement experiment_302_self_learning_benchmark.py (simulated+live_gpu paths, CaseMemory accumulation, ConstraintGenerator enrichment, improvement_delta signed reporting); 3841 total passed; update ops/changelog + status + traceability | ~20k |
| 4 | 2026-04-14T11:36:49Z | 2026-04-14T11:47:12Z | Exp 303 AMD XDNA NPU unblock: read Exp 292 script, results json, hardware-wishlist; check prereqs (ninja/openblas both missing); write 30 tests (TestExp303Schema, TestPrereqCheck, TestBuildOutcome, TestInferenceResult, TestNoFabricatedLatency); implement experiment_303_npu_unblock.py (prereq check→blocked_prereq, source build with 45-min timeout, wheel install, VitisAI inference benchmark, blocked_abi detection, wishlist updater); run experiment→blocked_prereq artifact; 3862 total passed; update ops/changelog + status | ~18k |
| 5 | 2026-04-14T11:53:12Z | 2026-04-14T11:59:41Z | Exp 304 HF publish: check credentials (CLI absent, Python API works — ianblenke/Carnot-EBM); write 24 tests (credential check, blocked schema, successful path, on-disk schema); implement experiment_304_hf_publish.py (CLI+API fallback, bypass Exp 293 internal CLI check, live upload); FCV artifact uploaded to Carnot-EBM/carnot-formal-claim-verifier-v1; exp66 skipped (no safetensors); 3886 total passed; update ops/changelog + status | ~15k |
| 6 | 2026-04-14T12:22:14Z | 2026-04-14T12:29:57Z | Exp 306 experiment template + batching harness: read Exp 294/258/302 patterns; write 54 tests (ExperimentTemplate init/setup/gpu/checkpoint/build_result/timeout, BatchedInferenceRunner grouping/timeout/logging, InferenceResult); implement experiment_template.py (ExperimentTemplate, BatchedInferenceRunner, InferenceResult, REQUIRED_RESULT_FIELDS); implement experiment_benchmark.py (20-question arithmetic benchmark, overhead_s=0.0001 < 0.5s target); run benchmark → results/experiment_306_results.json; 3975 total passed 54 skipped; update CLAUDE.md Experiment Template section; update ops/changelog + status | ~18k |
| 7 | 2026-04-14T13:02:26Z | 2026-04-14T13:32:19Z | Exp 307 JEPA retrain on real logits (MLP): add REQ-JEPA-004 + SCENARIO-JEPA-008/009 to spec; write 48 tests (extract_training_pairs, train_jepa_on_pairs, ONNX export via onnx.helper, run_experiment blocked/success, edge cases); implement experiment_307_jepa_real_training.py; 100% module coverage; update ops docs | ~16k |

---

## Session: 2026-04-14 Exp 294 GPU Stall Diagnosis + Apple Adversarial Baseline

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T09:09:51Z | 2026-04-14T09:18:54Z | Exp 294: GPU stall diagnosis + Apple adversarial baseline re-run — read exp_282/258 scripts, spec.md, add REQ-VERIFY-079/SCENARIO-101/102; 16 tests (prewarm health-check, timeout, artifact schema, accuracy bounds, logit saving, checkpoint resume, stall_at); implement model_prewarm() with concurrent.futures timeout, AppleBaselineRunner294 with pre-warm phase, 60s per-call timeout enforcement; adversarial review found missing timeout enforcement → fixed; 3523 total (16 new pass); ops/changelog + status updated | ~18k |
| 2 | 2026-04-14T09:29:18Z | 2026-04-14T09:35:06Z | Fix 12 retro test failures: stale operational_retro_2026_04_21.json missing experiments_in_scope/gpu_utilization_distribution/structural_action_taken/exp_per_hour fields; re-ran experiment_294_operational_retro.py to regenerate JSON; all 35 retro tests pass; 3535 total passed 99.11% coverage | ~8k |

---

## Session: 2026-04-14 Milestone Planning 2026.04.23

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T12:48:38Z | 2026-04-14T12:59:40Z | Plan milestone 2026.04.23 — read 11 project files (research-program, prd, architecture, status, changelog, research-complete, research-roadmap, change-proposals, conductor-log, research-references, hardware-wishlist), arxiv research (7 new papers via Explore agent), update research-references.md (8 new entries), create research-roadmap-vNEXT.md v29 (13 experiments across 4 phases, 3 gaps analysis, architecture diagram), create research-roadmap-next.yaml | ~40k |

## Session: 2026-04-14 Milestone Planning 2026.04.22

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T08:56:34Z | 2026-04-14T09:07:07Z | Plan milestone 2026.04.22 — read 10 project files, arxiv research (9 new papers), update research-references.md, create research-roadmap-v28.md (13 experiments across 4 phases), create research-roadmap-next.yaml | ~35k |

---

## Session: 2026-04-14 Exp 294 Operational Retro 2026.04.21

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T08:35:13Z | 2026-04-14T10:05:00Z | Exp 294: Operational retro milestone 2026.04.21 — 35 tests (retro artifact schema, carry-over computation, action item resolution, GPU utilization fields, structural root-cause), retro script (load 8 result files, wall-time from metrics.md, GPU distribution 11/0/2, 2/4 action items resolved, carry-over 50% ↓ from 100%), PROCESS-001 + PROCESS-002 story tickets created, results JSON written; 3519 total tests pass 99.11% coverage | ~30k |

---

## Session: 2026-04-14 Exp 293 HuggingFace Publish v0.2.0-research

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T08:16:50Z | 2026-04-14T08:23:53Z | Exp 293: HF publish carry-forward from 268 — 42 tests (incl. adversarial-review fixes: safetensors skip path, results-written-to-disk, create_tag, repo_ids in blocked), script with credential check, FCV ONNX (arithmetic+comparison opset 13) + Python module, model cards, upload_artifacts dry_run, results JSON; README + ops docs reconciled; 3484 total tests pass 99.11% coverage | ~35k |

---

## Session: 2026-04-14 Exp 292 AMD XDNA NPU VitisAI EP Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T08:00:14Z | 2026-04-14T08:11:05Z | Exp 292: AMD XDNA NPU VitisAI EP benchmark — 30 tests (blocked path), script with Path A (pre-built .so + ORT 1.20.1 downgrade) and Path B (source build, 45 min timeout); key finding: VitisAI EP must be compiled into ORT; blocked by ninja+openblas; reconciled docs; 3442 total tests pass 99.11% coverage | ~25k |

---

## Session: 2026-04-14 Exp 291 JEPA Apple Adversarial Retrain

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T07:52:01Z | 2026-04-14T07:54:00Z | Exp 291: Retrain JEPA predictor on Apple adversarial GPU data — 47 tests pass, TARGETS_MET (fast_path=0.500, TP=1.000, FP=0.000), ONNX exported to results/jepa_predictor_291.onnx | ~18k |

---

## Session: 2026-04-14 Exp 290 FpgaBackend vs CPU Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T07:17:01Z | 2026-04-14T07:30:00Z | Exp 290: FpgaBackend vs CPU benchmark — spec (REQ-SAMPLE-010, SCENARIO-SAMPLE-020/021/022), 27 tests, benchmark script (100/500/1000 spins, geometric vs linear schedule, LagONN penalty, 60s timeout, honest labeling), 3376 total tests pass 99.11% coverage | ~22k |
| 2 | 2026-04-14T07:33:11Z | 2026-04-14T07:36:05Z | Exp 290: ran benchmark script — CONFIRMED (geometric wins 3/3 sizes); updated docs/fpga-ising-design.md and ops/status.md with actual run numbers | ~8k |

---

## Session: 2026-04-14 Exp 289 FpgaBackend Quantum-Inspired Sparse Ising

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T07:01:20Z | 2026-04-14T07:14:15Z | Exp 289: FpgaBackend — quantum-inspired sparse Ising SamplerBackend. quantize_to_q88, sparsify_coupling, quantum_annealing_schedule, serialize_to_axi, _apply_lagrangian_penalty, FpgaBackend (PYNQ dispatch + geometric CPU fallback + LagONN), get_backend("fpga")→FpgaBackend, 47 tests 100% coverage fpga_backend.py, mypy clean, ruff clean, updated changelog/status/traceability | ~18k |

---

## Session: 2026-04-14 Exp 288 KV260 Bringup

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T06:49:07Z | 2026-04-14T06:55:52Z | Exp 288: KV260 FPGA overlay bring-up — spec (REQ-SAMPLE-009, SCENARIO-SAMPLE-018/019), 21 tests, bring-up script with env-var-first blocked path, spin ±1 validation, 60s hard timeout, 3302 total tests pass 99.11% coverage | ~20k |

---

## Session: 2026-04-14 Exp 284 Apple Analysis

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T05:44:59Z | 2026-04-14T05:52:13Z | Exp 284: Apple adversarial analysis and classification — spec (REQ-VERIFY-073–075, SCENARIO-VERIFY-088–092), 31 tests, analysis script with compute_delta/classify_result/compare_vs_exp235/answer_five_questions/build_artifact, INCONCLUSIVE (Exp 282/283 results missing), 3182 total tests pass | ~25k |

---

## Session: 2026-04-14 Exp 283 Apple Verify-Repair

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T05:29:11Z | 2026-04-14T05:39:13Z | Exp 283: verify-repair 12-cell benchmark on Apple adversarial corpus — spec (REQ-VERIFY-068–072, SCENARIO-VERIFY-084–087), 23 tests, VerifyRepairRunner with DualGPURunner at start, logits at 25/50/75/100% for JEPA, checkpoint every 10q, 60s timeout → partial artifact, primary criterion Δ(vr,ns)>Δ(vr,std), 3151 total tests pass | ~22k |

---

## Session: 2026-04-14 Exp 282 Apple Baseline GPU

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T05:15:06Z | 2026-04-14T05:22:16Z | Exp 282: GPU baseline inference on Apple adversarial corpus — 16 tests, AppleBaselineRunner with DualGPURunner wired at start, logits at 25/50/75/100%, checkpoint every 10q, 60s timeout → partial artifact with stall_at, 3128 total tests pass | ~18k |

---

## Session: 2026-04-14 Milestone 2026.04.21 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T04:45:29Z | 2026-04-14T04:57:15Z | Plan milestone 2026.04.21: read 11 key docs, arxiv scan (15 papers, 4 new), research-references.md updated, research-roadmap-vNEXT.md (v27) written, research-roadmap-next.yaml (14 experiments, 4 phases) written | ~35k |

---

## Session: 2026-04-14 Revalidation Sweep Summary

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T04:23:36Z | 2026-04-14T04:30:19Z | Revalidation sweep Exp 271-279: read all result JSONs, classified 6 CONFIRMED / 2 INCONCLUSIVE / 0 ruled out, wrote summary JSON, updated README/technical-report/index.html/research-studying.md, 3100 tests pass 99.10% coverage | ~20k |
| 2 | 2026-04-14T04:31:35Z | 2026-04-14T04:33:57Z | Summarize revalidation sweep: verified all docs already updated, ran pytest 3100 passed 99.10% coverage | ~8k |

---

## Session: 2026-04-14 Exp 278 Cross-Session Memory

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T03:54:42Z | 2026-04-14T04:02:45Z | Exp 278: Cross-session CaseMemory with live traces — 16 tests, populate from Exp 219-221 (94 entries), save/load session boundary, warm hit rate 100% vs cold 0%, FP rate 0%, 3084 total tests pass, 99.10% coverage. | ~18k |
| 2 | 2026-04-14T04:08:02Z | 2026-04-14T04:16:20Z | Exp 279: Adversarial semantic grounding — 50 pairs, stale_det=100%, fresh_det=0%, fp=20%, lift=+40pp, 3100 tests pass 99.10% coverage | ~14k |

---

## Session: 2026-04-14 Exp 274 KB Factual Live

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T02:31:13Z | 2026-04-14T02:41:43Z | Exp 274: FactualKBExtractor (embedded KB) on Gemma4-E4B-it responses — 66 tests pass, results JSON written; coverage 45% (target 40%), accuracy 100%. | ~18k |
| 2 | 2026-04-14T02:31:13Z | 2026-04-14T02:46:51Z | Fix failing tests: add 3 tests for generate_responses_with_gemma4 function (lines 668-675); 69 total tests pass, 100% coverage for exp274_kb_factual_live.py module. | ~8k |
| 3 | 2026-04-14T03:22:31Z | 2026-04-14T03:34:12Z | Exp 276: Full GSM8K with Z3+LLM+semantic extractors — script + 50 tests written; CI mode: Z3/LLM detect 4/5 wrong (80%), 0% FP; semantic 0% detection, 20% FP on arithmetic; combined 80% detection; all 3001/3002 suite tests pass. | ~22k |
| 4 | 2026-04-14T03:39:50Z | 2026-04-14T03:48:56Z | Exp 277: Combined signal benchmark — 30 HumanEval + 50 GSM8K (CI: 5+10); code+Z3+semantic for HE, Z3+LLM+semantic for GSM8K; interference_score computed; 66 new tests pass, 3067 total, 99.10% coverage. | ~28k |

---

## Session: 2026-04-14 arxiv Research Survey

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T00:28:28Z | 2026-04-14T00:29:54Z | arxiv survey: search 10 topics for recent 2025-2026 papers relevant to Carnot EBM milestone planning. | ~8k |

---

## Session: 2026-04-13 Exp 260 GPU Solver-Semantic Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T21:05:37Z | TBD | Exp 260: GPU-accelerated solver-semantic benchmark extending Exp 246/247 via DualGPUBenchmarkHarness — 25 tests written, all pass; live run launched with CARNOT_FORCE_CPU=0 on CUDA GPU 0 (RTX 3090). | TBD |

### Session Summary

- `scripts/experiment_260_solver_semantic_gpu.py` created, extending Exp 246 with DualGPUBenchmarkHarness
- 25 unit tests covering checkpoint resume, GPU harness integration, artifact schema, route summary aggregation
- All 25 tests pass in 0.38s
- Live run in progress: GPU verification passed (dual RTX 3090, ~24 GiB free each); CARNOT_FORCE_CPU=0 enables CUDA inference
- Existing Exp 246 checkpoints reused: all 6 Qwen GSM8K cells (200 cases × 3 modes) complete; Qwen constraint_ir baseline + verify_only complete; verify_repair in progress
- Observed: ~33s/case for constraint_ir verify_repair on GPU (multiple repair iterations per case)
- Status: **in progress** — run will complete to results/experiment_260_results.json

## Session: 2026-04-13 Exp 259 CUDA ORT Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T20:49:12Z | 2026-04-13T20:53:46Z | Exp 259: install onnxruntime-gpu, benchmark CUDA ORT for PredictiveVerifier gate — 14 tests written, all pass; CUDA ORT 47.3 µs/call (5.49× slower than CPU ORT due to kernel launch overhead on 9→1 linear gate); CPU NumPy 5.1 µs/call, CPU ORT 8.6 µs/call. | ~4.2k |
| 2 | 2026-04-13T21:02:59Z | 2026-04-13T21:03:54Z | Minimal doc sync for Exp 259 — appended 1-line changelog entry, ops/status.md experiment row, and 3 new SCENARIO-* rows to traceability.md; no content removed, only appended per doc-sync rules. | ~1.2k |

### Session Summary

- `pip install onnxruntime-gpu` successful; CUDAExecutionProvider + TensorrtExecutionProvider now available
- CPU NumPy (inference-only): 5.1 µs/call, 196,806 calls/s
- ONNX CPU ORT: 8.6 µs/call, 115,978 calls/s
- ONNX CUDA ORT: 47.3 µs/call, 21,142 calls/s (5.49× SLOWER than CPU ORT — expected for 9→1 linear gate)
- Key finding: CUDA kernel launch overhead dominates; GPU advantage appears at batch_size ≥ 32

## Session: 2026-04-13 Exp 258 Dual-GPU Harness

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T20:31:27Z | 2026-04-13T20:36:43Z | Exp 258: dual-GPU benchmark harness wiring DualGPURunner + ModelServer to Exp 218 interface — 35 tests written, all pass in 0.38 s. | TBD |

### Session Summary

- `scripts/experiment_258_dual_gpu_harness.py` created with `DualGPUBenchmarkHarness`, `ThroughputMeasurement`, `GPUAssignmentVerifier`, `write_harness_report`
- 35 unit tests covering GPU assignment, batching, memory cleanup, checkpoint interface, throughput target reporting
- All 35 tests pass; CARNOT_FORCE_LIVE=0 mock mode works without real GPU
- Target: ≤ 3 s/case per model (from 21 s/case on CPU in Exp 247)

## Session: 2026-04-13 Exp 257 Hardware Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T19:34:29Z | 2026-04-13T19:39:55Z | Exp 257: predictive-verifier hardware benchmark — 29 tests written, experiment script created, CPU (41.8 µs, 23.9k calls/s) and ONNX-CPU (5.8 µs, 171k calls/s, 7.1× faster) benchmarked; honest blockers emitted for CUDA ORT and AMD XDNA NPU. | TBD |

### Session Summary

- CPU NumPy gate: 41.8 µs/call, 23,938 calls/s
- ONNX CPU ORT: 5.8 µs/call, 171,032 calls/s (7.1× faster than full gate())
- CUDA ORT: BLOCKED — pip onnxruntime lacks CUDAExecutionProvider (need onnxruntime-gpu)
- AMD XDNA NPU: BLOCKED — VitisAI EP missing; Python 3.14 unsupported by AMD wheel
- 29 tests covering artifact labeling, export-path branching, blocker handling

## Session: 2026-04-13 Verify Test Suite

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T18:59:42Z | 2026-04-13T19:13:19Z | Fixed failing tests: verified all 2533 Python tests pass with 99.79% coverage (exceeds 99% requirement). Rust tests all passing. No code changes needed; previous changes are correct. | TBD |

### Session Summary

- All tests passing: 2533 passed, 2 skipped
- Coverage: 99.79% (exceeds 99% requirement)
- Rust: all tests pass, formatting check passes, clippy no warnings

## Session: 2026-04-13 REQ-PRED-001-004 Predictive Verifier Module

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T18:02:56Z | 2026-04-13T18:09:46Z | Added REQ-PRED-001..004 to spec; implemented `predictive_verifier.py` (feature extraction, calibrated gate, ONNX export, safetensors serialisation, duck-type jepa_predictor compatibility); wrote 48 tests covering features, gate routing, calibration, export, serialisation, pipeline integration. All 48 pass. | TBD |

### Session Summary

- 48 tests written and passing; `python/carnot/pipeline/predictive_verifier.py` (FEATURE_DIM=9, NumPy logistic gate, ONNX export, safetensors save/load, calibrate() from Exp 252 corpus rows, predict_embedding() duck-type compat) created.

## Session: 2026-04-13 Exp 252 Predictive Verification Corpus

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T17:31:48Z | 2026-04-13T17:40:15Z | Exp 252: wrote 10 tests (schema shape, determinism, provenance completeness, semantic+code coverage, memory-hit metadata, accepted-repair population), implemented `scripts/experiment_252_predictive_verification_corpus.py` building 683-record corpus from Exp 241/235/238/246/250 artifacts. All 10 tests pass. | TBD |

### Session Summary

- 10 tests written and passing; `data/research/predictive_verification_corpus_252.jsonl` (683 rows: 563 reasoning, 120 code) and `results/experiment_252_results.json` produced; 36 memory hits, 54 accepted repairs, 85 rejected.

## Session: 2026-04-13 Exp 251 Process Verification Comparison

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T17:13:59Z | 2026-04-13T17:18:22Z | Create results/experiment_251_results.json: process-verification comparison vs Exp 238 using completed Exp 250 checkpoints (30/30 cases, both models). Verdict: process adds rfwr detection but no pass@1 lift at gating stage. | TBD |

## Session: 2026-04-13 REQ-VERIFY-061/062 Process Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T15:58:23Z | 2026-04-13T16:03:39Z | Added REQ-VERIFY-061/062 to spec, implemented `process_verifier.py` with 6 defect kinds, added `verify_process_integrity` to `VerifyRepairPipeline`, wrote 29 tests covering reasoning/code-repair/IR/serialization/pipeline paths. All 29 pass. | TBD |

### Session Summary

- 29 tests written and passing; `python/carnot/pipeline/process_verifier.py` implemented; `VerifyRepairPipeline.verify_process_integrity` added additively.

## Session: 2026-04-13 Exp 248 Process Integrity Corpus

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T15:10:42Z | 2026-04-13T15:23:00Z | Exp 248: wrote 15 tests (schema shape, determinism, label coverage, provenance), implemented `scripts/experiment_248_process_integrity_corpus.py` with `classify_reasoning`/`classify_code` pure functions and deterministic JSONL builder. Corpus: 849 rows, all 5 process integrity labels covered across Exp 235 (reasoning) and Exp 238 (code). All 15 tests pass. | TBD |

### Session Summary

- 15 tests written and passing; `data/research/process_integrity_corpus_248.jsonl` (849 rows) and `results/experiment_248_results.json` produced.

## Session: 2026-04-13 Fix Test Failures (Exp 247 Provenance Count)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T14:46:54Z | TBD | Fixed test failure: `test_public_docs_disclose_current_provenance_inventory` was checking for 73 unverified artifacts but Exp 247 added a 74th. Updated README.md, docs/technical-report.md (2 locations), and docs/index.html to reflect 74 unverified + 91 total artifacts. Reran test suite. | TBD |

### Session Summary

- 1 failed test fixed: all tests now pass with 100% coverage.

## Session: 2026-04-13 VERIFY-058 Formal Claim Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T13:54:03Z | 2026-04-13T14:01:34Z | Formal claim verifier: added REQ-VERIFY-058/059 to spec, wrote 59 tests (test-first), implemented `formal_claim_verifier.py` with arithmetic/comparison/cardinality/set_membership/boolean_entailment routes + explicit abstain, integrated `verify_formal_claims` into `VerifyRepairPipeline` additively. All 59 tests pass. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

## Session: 2026-04-13 VERIFY-039 Learned Self-Learning Policy Compiler

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T07:47:58Z | 2026-04-13T08:09:25Z | VERIFY-039: extended `verifiable-reasoning` with `REQ-VERIFY-052` / `REQ-VERIFY-053` plus `SCENARIO-VERIFY-056` through `SCENARIO-VERIFY-059`, wrote `tests/python/test_self_learning_policy.py` first, implemented `python/carnot/pipeline/self_learning_policy.py` plus the public pipeline exports, reconciled traceability/status/changelog, and reran targeted 100% module coverage, changed-file Ruff + mypy + spec coverage, the full Python suite, the standard E2E trio, and reconciliation validation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

## Session: 2026-04-13 VERIFY-034 Exp 235 Live GSM8K Semantic Benchmark V2

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T04:19:42Z | 2026-04-13T04:49:26Z | VERIFY-034 / Exp 235: extended `verifiable-reasoning` with `REQ-VERIFY-048` / `REQ-VERIFY-049` plus `SCENARIO-VERIFY-050` / `SCENARIO-VERIFY-051`, wrote `tests/python/test_experiment_235_gsm8k_semantic_v2.py` first, implemented `scripts/experiment_235_gsm8k_semantic_v2.py`, reran targeted 100% script coverage, the full Python suite, spec coverage, Ruff, and the standard E2E trio, then completed the live Exp 235 GSM8K semantic rerun to `results/experiment_235_results.json` and reconciled traceability/status/changelog. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

## Session: 2026-04-13 VERIFY-033 Claim-Isolated Semantic Verifier V2

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-13T03:38:06Z | 2026-04-13T04:01:50Z | VERIFY-033: extended `verifiable-reasoning` with `REQ-VERIFY-046` / `REQ-VERIFY-047` plus `SCENARIO-VERIFY-047` / `SCENARIO-VERIFY-048` / `SCENARIO-VERIFY-049`, wrote `tests/python/test_semantic_verifier_v2.py` first, implemented `python/carnot/pipeline/semantic_verifier_v2.py` plus the additive `VerifyRepairPipeline` hook, reconciled spec/story/traceability/status/changelog, and reran targeted 100% module coverage, the full Python suite, lint/type/spec checks, E2E/integration checks, and reconciliation validation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

## Session: 2026-04-12 VERIFY-031 Packaged Code Verification

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T23:32:39Z | 2026-04-12T23:59:24Z | VERIFY-031: extended `code-verification` with `REQ-CODE-019` through `REQ-CODE-022` and `SCENARIO-CODE-016` through `SCENARIO-CODE-019`, wrote tests first for the packaged `verify_code()` API, `carnot verify-code`, `verify_code_with_pbt`, docs examples, and the generate-verify-repair E2E flow, implemented the new Python API/CLI/MCP surfaces plus docs, restored the final Python suite to `100.00%` coverage, reconciled traceability/status/changelog/test-results/e2e-plan, and reran the required validations. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

## Session: 2026-04-12 VERIFY-030 Code Verification Trace Learning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T22:48:50Z | 2026-04-12T23:21:56Z | VERIFY-030: extended `code-verification` with `REQ-CODE-016` / `REQ-CODE-017` / `REQ-CODE-018` plus `SCENARIO-CODE-014` / `SCENARIO-CODE-015`, wrote `tests/python/test_code_learning.py` first, implemented `python/carnot/pipeline/code_learning.py` plus the `carnot.pipeline` exports, reconciled traceability/status/changelog, and reran targeted 100% module coverage, Ruff, mypy, spec coverage, the full Python suite, and `tests/integration/test_full_pipeline.py`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 228 KV260 FPGA Ising Sampler Design

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T22:31:05Z | 2026-04-12T22:48:41Z | Exp 228: extended `training-inference` with `REQ-SAMPLE-005` / `REQ-SAMPLE-006` and `SCENARIO-SAMPLE-009` through `SCENARIO-SAMPLE-011`, wrote `tests/python/test_fpga_ising.py` first, implemented `python/carnot/samplers/fpga_ising.py` plus `get_backend("fpga")` wiring and the 4K-spin AXI-Lite design contract, documented the architecture in `docs/fpga-ising-design.md`, recorded the honest software-model benchmark in `results/experiment_228_results.json`, and reran targeted 100% module coverage, spec coverage, Ruff, mypy, the full Python suite, applicable E2E/integration checks, and reconciliation validation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 227 Seeded Qwen HumanEval PBT Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T21:59:44Z | 2026-04-12T22:23:20Z | Exp 227: extended `code-verification` with `REQ-CODE-015` and `SCENARIO-CODE-013`, wrote `test_experiment_227_qwen_pbt.py` first, implemented `scripts/experiment_227_qwen_pbt.py`, ran the live 30-problem Qwen3.5-0.8B HumanEval PBT benchmark on the exact Exp 208 cohort to `results/experiment_227_results.json`, reconciled spec/traceability/ops docs, and reran targeted 100% coverage for the new script, Ruff, spec coverage, the full Python suite, `tests/integration/test_full_pipeline.py`, and reconciliation validation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 226 Full HumanEval PBT Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T20:56:29Z | 2026-04-12T21:52:25Z | Exp 226: extended `code-verification` with `REQ-CODE-012` through `REQ-CODE-014`, wrote `test_experiment_226_pbt_humaneval_full.py` first, implemented `scripts/experiment_226_pbt_humaneval_full.py`, ran the full live 164-problem Gemma4-E4B-it HumanEval PBT benchmark to `results/experiment_226_results.json`, reconciled spec/traceability/ops docs, and reran targeted 100% coverage for the new script, Ruff, spec coverage, the full Python suite, `tests/integration/test_full_pipeline.py`, and reconciliation validation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 225 Dual-GPU Paired Inference Runner

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T19:47:00Z | 2026-04-12T20:34:48Z | Exp 225: extended `verifiable-reasoning` with `REQ-VERIFY-041` and `SCENARIO-VERIFY-042`, wrote `test_dual_gpu.py` plus the `model_loader` and Exp 218 harness parallel-dispatch assertions first, implemented `python/carnot/inference/dual_gpu.py` plus explicit `cuda:N` / `device_map="auto"` loading and the Exp 218 `--parallel` path, recorded the honest 10-question dual-GPU microbenchmark at `results/experiment_225_results.json`, reconciled specs/ops/story docs, and reran targeted diff-coverage, the full Python suite, spec coverage, Ruff, `tests/integration/test_full_pipeline.py`, CLI help, and reconciliation checks. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 224c TensorRT-LLM Backend

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T19:19:31Z | 2026-04-12T19:40:38Z | Exp 224c: extended `verifiable-reasoning` with `REQ-VERIFY-039` and `REQ-VERIFY-040`, wrote `test_tensorrt_backend.py` plus the warm-server preference assertions in `test_model_server.py` first, implemented `python/carnot/inference/tensorrt_backend.py` plus the `ModelServer` preference/export wiring and `cuda` extra update, generated blocked-status artifact `results/experiment_224c_results.json`, and reran targeted 100% coverage for the new module, the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Warm Server True Batched Forward Pass

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T18:56:19Z | 2026-04-12T19:11:08Z | Warm-server batching fix: tightened `verifiable-reasoning` so `REQ-VERIFY-036` / `REQ-VERIFY-037` require CUDA-requesting warm loads plus one padded `model.generate(...)` call per executed batch, wrote the new `test_model_server.py` assertions first, corrected `python/carnot/inference/model_server.py` and the shared helpers in `python/carnot/inference/model_loader.py`, closed `VERIFY-025`, and reran targeted 100% coverage, the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 224 Hypothesis-Backed PBT Code Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T17:43:30Z | 2026-04-12T18:13:46Z | Exp 224: extended `code-verification` with `REQ-CODE-009` through `REQ-CODE-011`, wrote `test_pbt_code_verifier.py` first, implemented `python/carnot/pipeline/pbt_code_verifier.py` plus the additive `VerifyRepairPipeline.verify_generated_code()` path, added the `hypothesis` dependency, reconciled traceability/status/changelog, and reran targeted 100% coverage for the new module, the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 223 Held-Out Live Self-Learning Replay

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T17:09:30Z | 2026-04-12T17:43:10Z | Exp 223: extended `verifiable-reasoning` with `VERIFY-033` through `VERIFY-035`, wrote `test_self_learning_replay.py` first, implemented `python/carnot/pipeline/self_learning_replay.py` plus `scripts/experiment_223_self_learning_replay.py`, generated `results/experiment_223_results.json`, reconciled traceability/status/changelog, and reran the required validation commands including targeted 100% coverage for the new module/script, the full Python suite, spec coverage, Ruff, mypy, `tests/integration/test_full_pipeline.py`, and `bash scripts/validate-reconciliation.sh`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 222 Live Trace Memory And Repair Guidance

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T16:44:47Z | 2026-04-12T17:09:21Z | Exp 222: extended `verifiable-reasoning` with `VERIFY-030` through `VERIFY-032`, wrote `test_live_trace_memory.py` first, implemented `python/carnot/pipeline/live_trace_memory.py` plus `scripts/experiment_222_live_trace_memory.py`, generated `results/experiment_222_results.json` and `results/constraint_memory_live_222.json`, reconciled traceability/status/changelog, and reran the required validation commands including targeted 100% coverage for the new module/script, the full Python suite, spec coverage, Ruff, mypy, `tests/integration/test_full_pipeline.py`, and `bash scripts/validate-reconciliation.sh`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 220 Live HumanEval Property Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T14:42:11Z | 2026-04-12T15:07:46Z | Exp 220: extended `verifiable-reasoning` with `VERIFY-028`, wrote tests first for HumanEval summary splits plus generation/repair traces, patched `scripts/experiment_218_live_dual_model_suite.py`, reran targeted 100% coverage plus the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`, then completed the live 50-problem/model HumanEval property benchmark to `results/experiment_220_results.json`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 219 Live GSM8K Semantic Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T12:41:55Z | 2026-04-12T14:28:18Z | Exp 219: extended `verifiable-reasoning` with `VERIFY-027`, wrote tests first for experiment-aware artifact metadata, GSM8K semantic summaries, trace serialization, and a live comma-only answer-extraction regression, patched `scripts/experiment_218_live_dual_model_suite.py`, reran targeted 100% coverage plus the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`, then completed the live 200-question/model GSM8K semantic benchmark to `results/experiment_219_results.json`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 218 Shared Dual-Model Live Benchmark Harness

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T12:15:49Z | 2026-04-12T12:35:06Z | Exp 218: extended `verifiable-reasoning` with `VERIFY-025` and `VERIFY-026`, wrote `test_experiment_218_live_dual_model_suite.py` first, implemented `scripts/experiment_218_live_dual_model_suite.py`, added deterministic cohort and shared-prompt-seed bookkeeping plus per-benchmark/model/mode checkpoints and a stable paired artifact schema, reconciled traceability/status/changelog, and reran the required verification commands including targeted 100% script coverage, the full Python suite, spec coverage, Ruff, mypy, `scripts/experiment_218_live_dual_model_suite.py --help`, `tests/integration/test_full_pipeline.py`, and `bash scripts/validate-reconciliation.sh`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 217 Prompt-Derived Property Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T11:37:05Z | 2026-04-12T12:07:10Z | Exp 217: extended `code-verification` with `REQ-CODE-006` through `REQ-CODE-008`, wrote `test_property_code_verifier.py` plus the HumanEval integration tests first, implemented `python/carnot/pipeline/property_code_verifier.py`, wired the additive property verifier into the Exp 208 execution path, reconciled traceability/status/changelog, and reran the required verification commands including targeted 100% coverage for the new module plus the HumanEval helper, the full Python suite, spec coverage, Ruff, mypy, and `tests/integration/test_full_pipeline.py`. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 216 Structured Reasoning Emission Path

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T11:19:05Z | 2026-04-12T11:36:28Z | Exp 216: extended `verifiable-reasoning` with `VERIFY-022` through `VERIFY-024`, wrote `test_structured_reasoning.py` plus gold fixtures first, implemented `python/carnot/pipeline/structured_reasoning.py`, added the additive `VerifyRepairPipeline.generate_structured_reasoning()` entry point, reconciled traceability/status/changelog, and reran the required verification commands including targeted 100% module coverage, the full Python suite, spec coverage, Ruff, mypy, and the full-pipeline integration test. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 215 Semantic Grounding Verifier

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T10:29:57Z | 2026-04-12T11:11:41Z | Exp 215: extended `verifiable-reasoning` with `VERIFY-020` and `VERIFY-021`, wrote `test_semantic_grounding.py` first, implemented `python/carnot/pipeline/semantic_grounding.py`, integrated additive semantic-grounding checks into `VerifyRepairPipeline`, reconciled traceability/status/changelog, and reran the required verification commands including targeted 100% module coverage, the full Python suite, spec coverage, Ruff, mypy, explicit E2E checks from `ops/e2e-test-plan.md`, and reconciliation. | TBD |

### Session Summary

- Authoritative token and cost extraction is currently blocked because the documented script `scripts/session-metrics.py` is not present in this checkout.
- Turn timing is recorded above; `Tokens (est)` remains `TBD` until the missing script or an equivalent replacement exists.

---

## Session: 2026-04-12 Exp 214 Semantic Failure Corpus

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T10:00:41Z | 2026-04-12T10:22:17Z | Exp 214: extended `verifiable-reasoning` with `VERIFY-018` and `VERIFY-019`, wrote `test_experiment_214_semantic_failure_corpus.py` first, implemented `scripts/experiment_214_semantic_failure_corpus.py`, generated `data/research/semantic_failure_corpus_214.jsonl` plus `results/experiment_214_results.json`, reconciled traceability/status/changelog, and reran the required verification commands including targeted 100% script coverage, the full Python suite, spec coverage, Ruff checks, and reconciliation. `ops/e2e-test-plan.md` has no model-training or cross-language item applicable to this deterministic corpus-generation workflow, so end-to-end verification for the task was the actual Exp 214 artifact generation command. | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 212 Typed Reasoning IR

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T09:29:29Z | 2026-04-12T09:52:30Z | Exp 212: extended `verifiable-reasoning` with `VERIFY-015` through `VERIFY-017`, wrote `test_typed_reasoning.py` first, implemented `python/carnot/pipeline/typed_reasoning.py`, wired the additive `VerifyRepairPipeline` hook, reconciled traceability/status/changelog, and reran the required validation commands including the full Python suite, targeted 100% module coverage, spec coverage, Ruff checks, reconciliation, and the explicit E2E checks from `ops/e2e-test-plan.md` | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 213 Monitorability Audit

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T08:36:59Z | 2026-04-12T09:21:46Z | Exp 213: extended `verifiable-reasoning` with `VERIFY-013`, wrote `test_experiment_213_monitorability_audit.py` first, implemented `scripts/experiment_213_monitorability_audit.py`, ran the live Qwen/Gemma monitorability audit over an 11-example Exp 211 subset in free-form / terse / structured modes, generated `results/experiment_213_results.json` plus `results/monitorability_policy_213.json`, reconciled traceability/status/changelog, and reran the required verification commands including the full Python suite, targeted 100% script coverage, spec coverage, reconciliation, and the explicit E2E checks from `ops/e2e-test-plan.md` | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 211 Constraint IR Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T08:08:12Z | 2026-04-12T08:29:58Z | Exp 211: extended `verifiable-reasoning` with `VERIFY-012`, wrote `test_experiment_211_constraint_ir_benchmark.py` first, implemented `scripts/experiment_211_constraint_ir_benchmark.py`, generated `data/research/constraint_ir_benchmark_211.jsonl` plus `results/experiment_211_results.json`, reconciled traceability/status/changelog, and reran the required verification commands including the full Python suite, explicit E2E tests, spec coverage, reconciliation, and 100% targeted coverage for the new script | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 210 Research Scan

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T07:03:16Z | 2026-04-12T07:22:54Z | Exp 210: extended the `research-reporting` spec plus `REPORT-002`, wrote `test_experiment_210_research_scan.py` first, implemented `scripts/experiment_210_research_scan.py`, refreshed `research-references.md` and `research-studying.md`, generated `results/experiment_210_results.json`, and reran the required verification commands including the full Python suite and 100% targeted coverage for the new script | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 209 Provenance Cleanup

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T06:31:36Z | 2026-04-12T06:53:38Z | Exp 209: added `research-reporting` spec + `REPORT-001`, wrote `test_experiment_209_cleanup.py` first, implemented `scripts/experiment_209_cleanup.py`, audited 66 result artifacts (5 validated live_gpu, 3 simulated, 58 unverified), rewrote README / technical report / landing page with provenance-aware claims, and reran the required verification commands including full Python suite and 100% targeted coverage for the new script | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 208 HumanEval Live Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T05:49:41Z | 2026-04-12T06:20:07Z | Exp 208: added `VERIFY-011`, implemented `humaneval_live_benchmark.py` plus 16 tests at 100% targeted coverage, added `scripts/experiment_208_humaneval_live_it.py`, reran `.venv/bin/pytest tests/python -q` at 100% suite coverage plus integration/lint/type checks, and completed the live 30-problem Gemma4-E4B-it HumanEval run; final result: baseline 5/30 (16.7%), verify-repair 6/30 (20.0%), Δ +3.3pp [0.0pp, +10.0pp], 1/25 failing baselines repaired | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 207 LLM vs Z3 Live Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T04:50:22Z | 2026-04-12T05:32:57Z | Exp 207: generalized `z3_live_benchmark.py` for named extractor comparisons, expanded `test_z3_live_benchmark.py` to 13 tests with 100% targeted coverage, added `scripts/experiment_207_llm_extractor_live.py`, reran `.venv/bin/pytest tests/python -q` at 100% coverage plus integration/lint/type checks, and completed the live Gemma4-E4B-it head-to-head run on the exact Exp 206 cohort; final result: LLM verify-only 90.0% with 1/91 false positives vs Z3 88.0% with 3/91 false positives, both 0/9 wrong-answer detections, both 91.0% verify-repair (Δ +0.0pp) | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 206 Z3 Live Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T03:52:14Z | 2026-04-12T04:31:34Z | Exp 206: added `z3_live_benchmark.py` + `experiment_206_z3_live.py`, wrote 9 tests with 100% `z3_live_benchmark.py` coverage, reran `.venv/bin/pytest tests/python -q` at 100% suite coverage, and completed the live Gemma4-E4B-it 100-question GSM8K benchmark; final live result: baseline 91%, Z3 verify-repair 91% (Δ 0.0pp), regex verify-repair 90% (Δ -1.0pp), Z3 false positives 3/91 vs regex 5/91, wrong-answer detection 0/9 for both; spec coverage still blocked by 11 pre-existing unrelated missing traces | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 205 LLM-as-Extractor

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T03:12:46Z | 2026-04-12T03:32:18Z | Exp 205: Added `LLMConstraintExtractor` with lazy `model_loader` hooks, canonical `CLAIM: a OP b = c` prompting, constant-energy claim terms, and per-response latency tracking; wrote 14 tests with 100% `llm_extractor.py` coverage plus an Exp 203 regression harness over the repo's current 3 wrong live Gemma cases and 3 correct showcases; `.venv/bin/pytest tests/python -q` passed at 100% coverage and `tests/integration/test_full_pipeline.py` passed; spec coverage / ruff / format-check / mypy remain blocked by pre-existing repo-wide failures | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 203 Extraction Autopsy

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T01:59:15Z | 2026-04-12T02:40:57Z | Exp 203: Added extraction-autopsy helper + live GPU script; ran seeded 20-question Gemma4-E4B-it GSM8K sample with full responses; final result 17/20 correct, 3 wrong; ArithmeticExtractor/VerifyRepairPipeline caught 0/3 wrong and flagged 3 correct-only violations; results saved to experiment_203_results.json | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-12 Exp 184 3B Model Scaling Study

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-12T00:09:39Z | 2026-04-12T01:30:18Z | Exp 184: 3B scaling study — ran Qwen3-4B on GPU0 (fallback from Qwen3.5-3B/Qwen3-3B not on HF); 200 standard GSM8K + 200 adversarial; baseline 63%/81.5%, repair 61%/68.5%; verify-repair HURTS on adversarial (-13pp, CI excludes zero); results saved to experiment_184_results.json | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-11 Exp 181 Definitive GSM8K Live GPU

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T21:25:47Z | 2026-04-11T21:41:57Z | Exp 181: Created scripts/experiment_181_gsm8k_live_gpu.py — definitive GSM8K 1319q × 2 models × 3 modes on RTX 3090 (no simulation); launched in background; GPU0 64% util, Qwen3.5-0.8B active | TBD |
| 2 | 2026-04-11T21:53:53Z | 2026-04-11T21:54:35Z | Exp 181 resume: read existing script (already complete), verified checkpoint at 100/1319 Qwen, relaunched — GPU0 67% util, 1824 MB VRAM, Qwen running live | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-11 Exp 180 Dual RTX 3090 GPU Baseline

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T21:03:39Z | 2026-04-11T21:12:38Z | Exp 180: Dual RTX 3090 GPU baseline — load time, VRAM, 50-prompt benchmark, GPU vs CPU speedup; Qwen 4.88x, Gemma 28x; fixed triton/Python 3.14 import issue | TBD |
| 2 | 2026-04-11T21:15:05Z | 2026-04-11T21:20:25Z | Fix test coverage gap (99.98% → 100%): added test for torch/transformers unavailable with CARNOT_FORCE_LIVE unset (lines 239-240 in model_loader.py); all 2484 tests pass with 100% coverage | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-11 Exp 178 Definitive Adversarial GSM8K

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T20:26:40Z | 2026-04-11T20:33:47Z | Exp 178: Definitive adversarial GSM8K N=400/variant paired permutation test; fixes Exp 162 underpowered permutation test (N=8→N=800 paired deltas); GOAL #5 ACHIEVED — number_swapped: perm p≈0 AND z p≈0; irrel_injected/combined not sig (logic errors, Ising can't catch); adv/ctrl ratio 1.19×; simulated inference | TBD |

---

## Session: 2026-04-11 Exp 176 Multi-Turn Factual Verification

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T19:55:13Z | 2026-04-11T20:05:22Z | Exp 176: Multi-turn factual reasoning verification combining FactualExtractor+ConstraintStateMachine+GlobalConsistencyChecker; 20 chains (10 consistent, 10 inconsistent); Mode A 0%, Mode B 60%, Mode C 100%; FP rate 0%; GlobalChecker adds +4 detections (all 4 numeric chains); _SingleArgPipeline wrapper added for agentic.propagate() compat | TBD |

---

## Session: 2026-04-11 Exp 173 Constraint Gen v2

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T19:02:48Z | 2026-04-11T19:12:38Z | Exp 173: NegationConstraint v2 (violation detection, not_all/no_A_are_B patterns) + CarryChainConstraint v2 (subtraction borrow, digit-count, negative-result); 58 new tests (100% generation.py coverage); cohort A negation recall 0%→100%; combined accuracy 0.9733 (+0.013 vs Exp141 0.96); dedup fix confirmed | TBD |
| 2 | 2026-04-11T19:17:55Z | 2026-04-11T19:31:54Z | Exp 174: LagONN (arxiv 2505.07179) — LagONN model, 46 tests 100% coverage, benchmark 20 SAT + 20 scheduling; scheduling: 0.5%→49.2% feasibility, 20/20 wins; SAT: mixed (encoding calibration needed); overall 23.2%→47.6% | TBD |
| 3 | 2026-04-11T19:35:21Z | 2026-04-11T19:47:10Z | Exp 175: AdaptiveKAN live verification tracking loop — adaptive_kan.py (KANConstraintModel base + AdaptiveKAN Tier-4), 45 tests 100% coverage, experiment 175 (3 AMR cycles: 500/1000/1500 verifications); AUROC 1.0 maintained across all 3 restructures; params 2310→2217 (-4%); ALL TARGETS PASS; 61.8s runtime | TBD |

---

## Session: 2026-04-11 Exp 171 Combined Signal Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T18:34:35Z | 2026-04-11T18:40:09Z | Exp 171: Combined signal benchmark — 200 questions (50 each: arithmetic/code/logic/factual), 5 configs; Config4 (Lookahead+Ising) best at 100% accuracy; Config2 (Ising-only) 80% (factual domain uncovered); Spilled energy (Config3/5) generates false positives at 0.5 nats threshold with V=1000; saved results/experiment_171_combined_results.json | TBD |

---

## Session: 2026-04-11 Exp 167 JEPA v3 Training

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T16:53:15Z | 2026-04-11T16:58:37Z | Exp 167: JEPA v3 training with symbolic logic features; created scripts/experiment_167_train_jepa_v3.py; combined 1500 pairs (800 arith + 200 code + 500 logic symbolic); trained AdamW 200ep patience=20; logic AUROC 0.479→0.946, macro 0.659→0.932; both targets MET; 46/46 tests pass; saved results/jepa_predictor_v3.safetensors | TBD |

---

## Session: 2026-04-11 Exp 164 HuggingFace Publishing

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T15:29:40Z | 2026-04-11T15:34:14Z | Exp 164: HuggingFace publishing — created scripts/experiment_164_hf_publish.py; uploaded guided-decoding-adapter, 3 constraint-propagation models, JEPA v2; updated 16 per-token EBM READMEs with pip install note; 5/5 uploads verified, 16/16 READMEs updated | TBD |

---

## Session: 2026-04-11 Exp 163 Full HumanEval Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T15:08:48Z | 2026-04-11T15:19:34Z | Exp 163: Full HumanEval benchmark (164 problems); created scripts/experiment_163_humaneval_full.py; loaded real HumanEval from HuggingFace; simulation mode (CARNOT_SKIP_LLM=1); baseline 68.9% [61.6%,75.6%], repair 100% [100%,100%], Δ+31.1%; 4.7s runtime; results saved to experiment_163_results.json | TBD |

---

## Session: 2026-04-11 Exp 162 Powered Adversarial GSM8K

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T14:20:23Z | 2026-04-11T14:40:49Z | Exp 162: Powered adversarial GSM8K (N=200/variant, 10k permutations, two-proportion z-test); Goal #5 definitive; live CPU inference killed (~17s/q → 7hr est); simulation fallback; z-test p=0.017 SIGNIFICANT; perm-test p=0.429 (structural underpowering); ratio 1.41×; script + results saved | TBD |
| 2 | 2026-04-11T14:44:41Z | 2026-04-11T14:45:59Z | Minimal doc updates: append Exp 162 to ops/status.md (High Priority section), _bmad/traceability.md (research validation table); changelog already has entry from Exp 162 script | TBD |

---

## Session: 2026-04-11 Exp 161 Full GSM8K Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T14:05:56Z | 2026-04-11T14:11:24Z | Exp 161: Full GSM8K (1319 questions) × 2 models × 3 modes + 95% bootstrap CIs; real GSM8K dataset loaded; simulation fallback (CARNOT_SKIP_LLM=1); Qwen3.5: +13.8% [+12.0%,+15.7%], Gemma4: +10.7% [+9.1%,+12.4%]; Goal #6 PARTIAL (real data, simulated inference); saved results/experiment_161_results.json | TBD |

---

## Session: 2026-04-11 Exp 158 FactualExtractor

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T13:20:24Z | 2026-04-11T13:34:57Z | Exp 158: FactualExtractor (Wikidata SPARQL) — factual_extractor.py + 69 tests (100% cov) + AutoExtractor enable_factual_extractor= param; benchmark: coverage=96.0% (target >30% ✓), accuracy=83.3%; results/experiment_158_results.json | TBD |

---

## Session: 2026-04-11 Exp 157 Spilled Energy Pre-Filter

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T12:56:05Z | 2026-04-11T13:07:17Z | Exp 157: SpilledEnergyExtractor (arxiv 2602.18671 ICLR 2026) — spilled_energy.py + 33 tests (100% cov) + AutoExtractor logits= param; benchmark AUROC=1.000 (target >0.60 ✓); coverage 100% vs NLExtractor 60%; results/experiment_157_results.json | TBD |

---

## Session: 2026-04-11 Exp 156 JEPA Fast-Path v2

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T11:46:44Z | 2026-04-11T11:49:59Z | Exp 156: JEPA fast-path v2 validation — v2 predictor vs v1 at thresholds 0.3/0.5/0.7; target NOT MET (no threshold achieved <2% degradation); best: t=0.5 → 52.8% fast-path, 10.2% degradation (v1: 95.4%/19.8%); code domain still dominates errors (42/51 at t=0.5); root cause: code pipeline fast-paths entire domain; saved results/experiment_156_results.json | TBD |

---

## Session: 2026-04-11 Planning Agent — Milestone 2026.04.11

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T10:12:16Z | 2026-04-11T10:19:32Z | Plan milestone 2026.04.11: read 14 source files + arxiv research; wrote research-roadmap-vNEXT.md (v16) + research-roadmap-next.yaml (12 experiments, 4 phases); updated research-references.md with 5 new arxiv papers; 3 biggest gaps: JEPA multi-domain fix, factual extractor, live eGPU benchmarks | TBD |

---

## Session: 2026-04-11 Exp 152 ContinualGibbs

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T09:28:39Z | 2026-04-11T09:37:39Z | Exp 152: ContinualGibbsModel (orthogonal gradient continual learning); 29 tests, 100% module coverage; benchmark vs Ising/LNN (Exp 116) — ContinualGibbs 100% step-5 accuracy (target >80% met); results/experiment_152_results.json | TBD |
| 2 | 2026-04-11T09:52:26Z | 2026-04-11T10:02:54Z | Exp 153: KAN adaptive mesh refinement — compute_edge_curvature() + refine(threshold=1.5); AUROC 0.875→0.875 (delta=0.000, ✓), params 2310→2281 (-1.3%, ✓); 36 knots added, 65 removed; high-curv=domain×numeric edges, low-curv=within-group edges | TBD |

---

## Session: 2026-04-11 Constraint Propagation Model Export

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T08:58:24Z | 2026-04-11T09:11:35Z | Create exports/constraint-propagation-models/ (arithmetic AUROC=0.997, logic 1.0, code 0.867); python/carnot/inference/constraint_models.py (IsingConstraintModel + ConstraintPropagationModel with from_pretrained/save_pretrained); scripts/export_constraint_models.py (training + export); 3 model cards + collection README; 52 tests passing, constraint_models.py 100% coverage | TBD |

---

## Session: 2026-04-11 Exp 149 TruthfulQA Factual Coverage Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T07:45:33Z | 2026-04-11T07:49:58Z | Create + run scripts/experiment_149_truthfulqa.py; TruthfulQA 196q balanced 7 cats; overall coverage 43.4%; KB adds 0%; covered q: accept 100%, reject 0% (shallow extraction); top-1 missing: world_knowledge (8.1% gain); recommend FactualWorldKnowledgeExtractor; results/experiment_149_results.json | TBD |

---

## Session: 2026-04-11 Pre-research test suite check

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T07:32:37Z | 2026-04-11T07:43:14Z | Run full test suite to verify pre-research baseline — 2041 passed, 1 skipped, 0 failures, 99.26% coverage (≥99% threshold met) | TBD |

---

## Session: 2026-04-11 Exp 147 Apple GSM8K Adversarial

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:52:33Z | 2026-04-11T05:07:44Z | Create scripts/experiment_147_apple_gsm8k.py; 3-mode eval (baseline/verify/verify-repair) × 4 adversarial variants × 2 models; Qwen number-swapped: baseline 46% → VR 73% (+27pp); Gemma number-swapped: 53% → 77.5% (+24.5pp); control deltas: +10pp/+13pp; hypothesis direction supported (num-swap delta >> control delta); permutation test p=0.463 (N too small); results/experiment_147_results.json | TBD |

---

## Session: 2026-04-11 Exp 146 AMD XDNA NPU Latency Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:41:15Z | 2026-04-11T04:44:27Z | Create scripts/experiment_146_npu.py + python/carnot/samplers/npu_backend.py; detect NPU HW (present: /dev/accel0, amdxdna loaded) vs SW (AMDXDNAExecutionProvider absent from std onnxruntime); export JEPA MLP to ONNX; CPU benchmark p50=0.005ms p99=0.009ms; NPU blocked — needs conda install -c amd onnxruntime-vitisai; results/experiment_146_npu_results.json | TBD |

---

## Session: 2026-04-11 Exp 145 JEPA Fast-Path Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:17:02Z | 2026-04-11T04:23:55Z | Add JEPA fast-path gate to VerifyRepairPipeline.verify() (jepa_predictor=, jepa_threshold=, mode/skipped fields on VerificationResult); 8 new tests 100% coverage; create + run scripts/experiment_145_jepa_fastpath.py; threshold=0.3: 38% fast-path (miss), 11.6% degradation (miss); threshold=0.5: 95.4% fast-path (pass), 19.8% degradation (miss); targets not met — predictor AUROC 0.57 insufficient; results/experiment_145_results.json | TBD |

---

## Session: 2026-04-11 Exp 142 Combined Learning Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:03:52Z | 2026-04-11T04:09:25Z | Create scripts/experiment_142_combined_learning.py — 4-way benchmark (Baseline/Tier1/Tier2/Combined) on 500 questions; Tier2 beats Tier1: YES (+7%, 71.75%→78.75%); Combined≈Tier2 (no Tier1 additive gain); 100% of Tier2 gains from new constraints; results/experiment_142_results.json | TBD |

---

## Session: 2026-04-11 Exp 141 Constraint Generation from Memory

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:45:37Z | 2026-04-11T03:55:16Z | Create generation.py (ConstraintGenerator, CarryChainConstraint, BoundConstraint, NegationConstraint); extend AutoExtractor.extract(memory=); 62 tests 100% coverage; Exp 141 benchmark: static=0.85→memory=0.96 (+0.11 delta, hypothesis MET); adversarial review found+fixed dedup bug | TBD |

---

## Session: 2026-04-11 Exp 144 JEPA Predictor Training

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:32:26Z | 2026-04-11T03:37:53Z | Create python/carnot/pipeline/jepa_predictor.py (JEPAViolationPredictor MLP 256→64→32→3), tests/python/test_jepa_predictor.py (36 tests, 100% coverage), scripts/experiment_144_train_jepa.py; train on Exp 143 pairs: arithmetic AUROC=0.7126, macro AUROC=0.5709 (code/logic AUROC=0.5 — no positives in data); model saved to results/jepa_predictor.safetensors (73.1 KB) | TBD |

---

## Session: 2026-04-11 Exp 143 JEPA Training Pair Collection

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:09:05Z | 2026-04-11T03:12:06Z | Create scripts/experiment_143_collect_pairs.py — mines result logs (0 pairs found), generates 200 synthetic arithmetic pairs via AutoExtractor+VerifyRepairPipeline, embeds 4 prefix ratios with RandomProjection(256-dim, seed=42); 800 total pairs, 33.5% positive rate; saved to results/jepa_training_pairs.json | TBD |

---

## Session: 2026-04-11 Exp 140 Constraint-Projection Guided Latency

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:50:09Z | 2026-04-11T03:00:29Z | Create scripts/experiment_140_guided_latency.py — adds project_logits() to EnergyGuidedSampler; benchmarks constraint-projection at batch sizes 1/8/32 (p50=0.405/1.284/4.056 ms); GSM8K accuracy baseline=56% penalty=64% projection=60%; success criterion PASS (0.405ms < 5ms); results in results/experiment_140_results.json | TBD |

---

## Session: 2026-04-11 Exp 139 ArXiv Scan

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:30:45Z | 2026-04-11T02:34:00Z | Create scripts/experiment_139_arxiv_scan.py — scans arXiv for 8 query topics, selects top 10 papers, annotates with Carnot relevance, proposes EXP-140/141/142; appends 10 new papers to research-references.md; results in results/experiment_139_results.json | TBD |

---

## Session: 2026-04-11 Exp 138 Guided Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:20:12Z | 2026-04-11T02:24:15Z | Create scripts/experiment_138_guided_benchmark.py — 3-task guided decoding benchmark (GSM8K 200, HumanEval 50, TruthfulQA 100); 4 modes; latency profiling; results saved to results/experiment_138_results.json | TBD |

---

## Session: 2026-04-11 Doc update Exp 137

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:16:21Z | 2026-04-11T02:17:11Z | Update ops/status.md + ops/changelog.md for Exp 137 (HF guided decoding adapter export); changelog was already written by conductor; added status.md header update + new HuggingFace Guided Decoding Adapter Export section | TBD |

---

## Session: 2026-04-11 guided-decoding-adapter export

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:09:32Z | 2026-04-11T02:14:00Z | Create exports/guided-decoding-adapter/ — HuggingFace-publishable artifact for GuidedDecoder; added GuidedDecoder.from_pretrained() API to guided_decoding.py; 7 new tests all pass; example.py verified | TBD |

---

## Session: 2026-04-11 Exp 136 Cross-Session Memory

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:57:14Z | 2026-04-11T02:02:12Z | Create scripts/experiment_136_cross_session.py — 3-session cross-session memory test (200 arith S1, 200 arith S2 no-mem vs mem, 200 mixed S3); all 4 hypotheses pass: memory accumulates (115 patterns), S2 hint delta +1.0/q, repair speedup 1.43x, domain specificity (logic/code get 0 hints); 0.5s wall-clock | TBD |

---

## Session: 2026-04-11 Exp 134 Online Learning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:21:28Z | 2026-04-11T01:36:32Z | Create scripts/experiment_134_online_learning.py — online learning demo with soft weighted-score verifier + NoisyHeuristicExtractor + ground-truth tracker recording; fixed=67.6%, adaptive=97.0%, delta=+29.4% overall; at q200 delta=+42.0% (target met) | TBD |

---

## Session: 2026-04-11 AdaptiveWeighter (Tier 1 Self-Learning)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:07:05Z | 2026-04-11T01:12:37Z | Create python/carnot/pipeline/adaptive.py (AdaptiveWeighter: from_tracker/apply_to_pipeline, run_comparison, ComparisonResult) + modify verify_repair.py to use _adaptive_weights + tests/python/test_adaptive.py (23 tests, 100% coverage, 1895 total pass) | TBD |

---

## Session: 2026-04-11 ConstraintTracker (Tier 1 Self-Learning)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:51:06Z | 2026-04-11T00:59:08Z | Create python/carnot/pipeline/tracker.py (ConstraintTracker: record/precision/recall/stats/save/load/merge) + integrate into VerifyRepairPipeline.verify(tracker=) + tests/python/test_tracker.py (28 tests, 100% coverage, 1872 total pass) | TBD |

---

## Session: 2026-04-11 Exp 121 Adversarial Verify-Repair

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:28:47Z | 2026-04-11T00:32:25Z | Run experiment_121_adversarial_verify_repair.py (CARNOT_SKIP_LLM=1 simulation); results/experiment_121_results.json created (17KB); Qwen3.5-0.8B hypothesis p=0.005 (supported), Gemma4-E4B-it p=0.290 (not significant) | TBD |
| 2 | 2026-04-11T00:35:58Z | 2026-04-11T00:36:29Z | Update docs for Exp 130 adversarial verify-repair: add Exp 121 entry to _bmad/traceability.md | TBD |
| 3 | 2026-04-11T00:40:11Z | 2026-04-11T00:43:33Z | Exp 131: Create adversarial writeup script; generates comparison tables (per-variant/mode/model), bootstrap CIs, appends Section 18 to docs/technical-report.md, saves experiment_131_results.json | TBD |

---

## Session: 2026-04-11 LiquidConstraintModel (lnn.py)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:12:49Z | 2026-04-11T00:20:55Z | Create python/carnot/models/lnn.py (LiquidConstraintModel: MLP-driven dJ/dt ODE, step(), energy(), reset(), train() BPTT) + tests/python/test_lnn.py (31 tests, 100% coverage, 1844 total pass) | TBD |

---

## Session: 2026-04-10 Exp 126 Agent Rollback

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:41:35Z | 2026-04-10T23:48:24Z | Create scripts/experiment_126_agent_rollback.py: 20 4-step math problems with error propagation, CSM rollback on violation detection; step-3 errors 100% detected, step-2 errors 0% detected; overall 50% improvement (0%→50% accuracy) | TBD |

---

## Session: 2026-04-10 ConstraintStateMachine

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:31:09Z | 2026-04-10T23:34:18Z | Create python/carnot/pipeline/state_machine.py (ConstraintStateMachine, StepResult, rollback, history, verified_facts, pending_facts) + 26 tests, state_machine.py 100% coverage | TBD |

---

## Session: 2026-04-10 Exp 122 Adversarial Error Analysis

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:14:56Z | 2026-04-10T23:23:33Z | Create Exp 122 adversarial error analysis: error taxonomy, Carnot detection rates per type, energy-prediction ROC (n_violations AUC=0.677), irrelevant-number extraction robustness; results saved | TBD |

---

## Session: 2026-04-10 Exp 120 Adversarial Baseline

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:59:18Z | 2026-04-10T22:13:37Z | Create Exp 120 adversarial baseline: LLM accuracy on 4 adversarial GSM8K variants, simulation mode (models too slow on CPU), results saved | TBD |

---

## Session: 2026-04-10 Robust Model Loader

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:41:01Z | 2026-04-10T21:51:47Z | Create carnot.inference.model_loader — robust HF model loading with memory check, OOM retry, CARNOT_FORCE_LIVE; 35 tests, 100% coverage, 1787 full suite pass | TBD |

---

## Session: 2026-04-10 Exp 119 Adversarial GSM8K

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:00:56Z | 2026-04-10T21:04:56Z | Create Exp 119 adversarial GSM8K (Apple 2410.05229 repro): 4 datasets × 200q, all 40 spot-checks pass | TBD |

---

## Session: 2026-04-10 Exp 118 HuggingFace Publish v12 Artifacts

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:55:35Z | 2026-04-10T20:56:17Z | Update changelog/status docs for Exp 118 HuggingFace v12 artifact publish | TBD |

---

## Session: 2026-04-10 Exp 117 Full v12 Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:29:53Z | 2026-04-10T20:37:00Z | Create Exp 117 full 4-mode v12 benchmark (2000 evaluations), run comparison vs v10, guided gen wins 10/10 cells | TBD |

---

## Session: 2026-04-10 Exp 116 LNN Adaptive Constraint Model

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:15:29Z | 2026-04-10T20:21:53Z | Create LNNConstraintModel (LTCN-based adaptive EBM), 22 tests (100% module cov), Exp 116 synthetic chain comparison vs Ising | TBD |

---

## Session: 2026-04-10 Exp 113 FactualKBExtractor

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T18:15:15Z | 2026-04-10T18:27:05Z | Create FactualKBExtractor with 5000-fact KB, 78 tests (100% cov), register in AutoExtractor | TBD |

---

## Session: 2026-04-10 Exp 112 Embedding Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T17:53:16Z | 2026-04-10T17:59:11Z | Create fast_embedding.py (5 strategies + protocol), experiment_112 script, run benchmark, update ops | TBD |

---

## Session: 2026-04-10 Exp 110 Guided Decoding

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T17:10:32Z | 2026-04-10T17:19:14Z | Create EnergyGuidedSampler, 22 tests (100% cov), Exp 110 on 50 GSM8K problems, alpha sweep [0.1–2.0] | TBD |

---

## Session: 2026-04-10 Exp 102 Latency Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T04:54:17Z | 2026-04-10T05:01:01Z | Create Exp 102 latency benchmark, run on CPU, save results + summary | TBD |

---

## Session: 2026-04-10 Exp 93 Multi-Model Comparison

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T01:59:33Z | 2026-04-10T02:07:00Z | Create Exp 93 multi-model comparison script, run benchmark, update ops | TBD |

---

## Session: 2026-04-09 Exp 57 Verify-Repair Loop

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-09T14:08:06Z | 2026-04-09T14:12:40Z | Create Exp 57 verify-repair loop script, run E2E with live LLM | TBD |

---

## Session: 2026-04-07 Research Roadmap v5 + Nemotron Analysis

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-07T18:59:33Z | 2026-04-07T19:03:43Z | Analyze Nemotron 3 Super paper, fold findings into roadmap v5 | TBD |
| 2 | 2026-04-07T19:08:49Z | 2026-04-07T19:15:52Z | Restructure roadmap v5 as weight-first (label-free) research program | TBD |
| 3 | 2026-04-07T19:21:22Z | 2026-04-07T19:32:00Z | Download Mixtral-8x7B, write Exp 32+33 scripts, update ops docs | TBD |

---

## Session: 2026-04-06 Documentation UI Modernization

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-06T18:00:00Z | 2026-04-06T18:05:00Z | Elevate docs/index.html to a premium aesthetic (glassmorphism, animations) | TBD |

---

## Session: 2026-04-06 GEMINI.md Initialization

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-06T16:47:49Z | TBD | Initialize GEMINI.md based on CLAUDE.md; adapt project mandates | TBD |

---

## Session: 2026-04-05 Hallucination Direction

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-05T05:18:58Z | TBD | Implement hallucination_direction.py with tests, exports, specs | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-03 Bootstrap

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-03T14:20:41Z | TBD | Initial project bootstrap: BMAD, specs, Rust workspace, Python package, pre-commit | TBD |
| 2 | 2026-04-10T20:45:20Z | 2026-04-10T20:52:49Z | Publish KAN + guided decoding adapter as HF-ready artifacts in models/constraint-verifier-v2 | 7m29s |
| 3 | 2026-04-11T11:23:30Z | 2026-04-11T11:31:54Z | Exp 155: Retrain JEPA v2 on multi-domain data; generate v2 pairs, train with weighted BCE + early stopping, evaluate vs v1 | 8m24s |
| 4 | 2026-04-11T16:44:07Z | 2026-04-11T16:45:10Z | Doc updates for Exp 166: append changelog entry, traceability row, verify REQ-JEPA-001 + SCENARIO-JEPA-LOGIC-001 | 1m3s |
| 5 | 2026-04-11T18:26:42Z | 2026-04-11T18:28:48Z | Exp 170: create real-logits benchmark (100 Q, simulated fallback — torch not installed); SpilledEnergy AUROC=1.000, LookaheadEnergy AUROC=1.000, optimal α=0.0; results saved | 2m6s |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*
| 6 | 2026-04-11T20:36:12Z | 2026-04-11T20:37:59Z | Minimal doc updates for Exp 178 (append entries to changelog.md, status.md, traceability.md); Goal #5 ACHIEVED | 1m47s |
| planning | 2026-04-13T19:58:08Z | 2026-04-13T20:11:45Z | Plan milestone 2026.04.19: read all context files, arxiv research, wrote research-roadmap-vNEXT.md + research-roadmap-next.yaml + research-references.md update | 13m37s |
| exp281 | 2026-04-14T05:01:31Z | 2026-04-14T05:09:44Z | Exp 281: Apple adversarial GSM8K dataset generator — spec (REQ-VERIFY-063, SCENARIO-VERIFY-078/079), 12 tests, dataset generator (400 rows, number_swap+irrelevant_sentence), 3112 tests pass | 8m13s |
| exp296 | 2026-04-14T09:57:16Z | 2026-04-14T10:04:18Z | Exp 296: Apple adversarial analysis v2 — tests, script, artifact; classification INCONCLUSIVE (Exps 294/295 missing); 45 tests added, 3609 total pass, 99.11% coverage | 6m62s |
| exp298 | 2026-04-14T10:29:25Z | 2026-04-14T10:30:16Z | Minimal doc updates for Exp 298 (PrefillUncertaintyProbe): verified changelog + status already updated by commit, appended 3 rows to traceability.md (REQ-VERIFY-080, SCENARIO-VERIFY-103/104) | 0m51s |
| exp291-fpga | 2026-04-14T12:06:46Z | 2026-04-14T12:20:21Z | Exp 291 FPGA RTL: 128-spin Verilog RTL (ising_sampler_v1.v), Python behavioral sim (simulate_ising_sampler.py), 36 tests passing, hardware/kv260/README.md; REQ-SAMPLE-011, SCENARIO-SAMPLE-023/024 | 13m35s |
| exp308 | 2026-04-14T15:31:56Z | 2026-04-14T15:40:13Z | Exp 308: JEPA gate benchmark — all code pre-written; fixed logit_mean dim (32→8 for Exp291 ONNX); ran 28 tests (pass); ran benchmark (TARGET NOT MET: skip_rate=0.0, Exp307 model missing); updated ops docs | 8m17s |
| exp316 | 2026-04-14T20:29:35Z | 2026-04-14T20:29:55Z | Minimal doc updates for Exp 316: append changelog entry (execution status) and status row (in progress) | 0m20s |
| exp326 | 2026-04-15T02:03:56Z | 2026-04-15T02:27:12Z | Exp 326: DualGPUMonitor (RETRO-002 + RETRO-003) — dual_gpu_monitor.py, GPUProcessInfo dataclass, check_dual_gpu_health(), setup_gpu() additive gpu_monitor_results key, 32 tests pass, 0 regressions (4784+79 pre-existing pass/skip), REQ-INFRA-003/004 | 23m16s |
| exp327 | 2026-04-15T02:29:40Z | 2026-04-15T02:54:55Z | Exp 327: Pre-experiment dependency audit (NEW-002) | TBD |
| exp335 | 2026-04-15T06:01:06Z | 2026-04-15T06:08:22Z | Exp 335: AMD XDNA NPU build 4th retry — spec (SCENARIO-EXP303-E/F), 4 check fns, prereq_status(), prereq_changes_vs_exp314(), attempt_ort_source_build(); 50 tests pass, 11 skip; blocked_prereq (ninja+openblas still missing) | 7m16s |
| exp348 | 2026-04-15T14:34:35Z | 2026-04-15T14:35:30Z | Minimal doc updates for Exp 348 (SinkProbe attention-sink pre-filter): appended 1 changelog entry, 1 status row, 5 traceability rows (REQ-VERIFY-086/087, SCENARIO-VERIFY-113/114/115) | 0m55s |

| 2026-04-15T06:21:22Z | 2026-04-15T06:43:30Z | Exp 337 retro: milestone 2026.04.24 operational retrospective | 22m | — |
| exp355 | 2026-04-15T17:17:54Z | 2026-04-15T17:19:43Z | Exp 355: Adversarial GSM8K benchmark — live GPU execution harness; run tests (51 pass); run script (simulated mode, honest_verdict=blocked_simulated); update ops docs | 1m49s |
| exp358 | 2026-04-15T17:46:53Z | 2026-04-15T17:58:27Z | Exp 358: Comparative extraction benchmark — extraction_benchmark.py + test (33 pass) + experiment script + spec/traceability/ops docs | 11m34s |
| 2026-04-15T19:54:06Z | 2026-04-15T19:55:16Z | Doc updates for Exp 361 (changelog, status) | 1 min 10 sec | — |
| planning-32 | 2026-04-16T16:27:40Z | 2026-04-16T16:37:41Z | Plan milestone 2026.04.32: read 10 context files, arxiv research (12 papers), wrote research-roadmap-vNEXT.md (v38) + research-roadmap-next.yaml (12 experiments) + research-references.md updates | 10m01s |
| exp433 | 2026-04-17T04:44:34Z | 2026-04-17T05:10:00Z | Exp 433: SpilledEnergyDetector — per-token logit-discrepancy hallucination signal (arXiv 2602.18671); SpilledEnergyToken + SpilledEnergyDetectorResult + compute_detector_spilled_energy + SpilledEnergyDetector + Tier 0 in ThreeTierPipeline; 28 new tests pass | ~25m |
| exp435 | 2026-04-17T05:43:29Z | 2026-04-17T05:52:05Z | Exp 435: AMD XDNA NPU Unblock — 5th attempt + IRON toolchain probe; REQ-PRED-005 + SCENARIO-EXP303-G; 50 tests pass; blocked_prereq escalation | ~8.5m |
| planning-33 | 2026-04-17T07:24:00Z | 2026-04-17T07:25:47Z | Plan milestone 2026.04.33: 11 context files read, 5 new papers added to research-references.md, research-roadmap-vNEXT.md (v39) + research-roadmap-next.yaml (13 exps: 437-449) created | ~90m (split session) |
| exp437 | 2026-04-17T07:37:18Z | 2026-04-17T07:50:22Z | Exp 437: LongRunBenchmarkExecutor — RETRO-026 closed; long_run_executor.py + 25 tests 100% coverage + experiment script + spec/ops docs | 13m04s |
| exp449 | 2026-04-18T09:41:35Z | 2026-04-18T09:54:14Z | Exp 449: Milestone 2026.04.33 retrospective — MilestoneRetro2026_04_33 dataclass; SCENARIO-RETRO-033; RETRO-024/026 CLOSED; RETRO-028/029/030/031 opened; 75 tests pass; all reconciliation docs updated | ~13m |
| exp453 | 2026-04-18T16:58:30Z | 2026-04-18T17:19:15Z | Exp 453: VeriCoT Step Validator — FOL formalization + Z3 UNSAT for IT model CoT; 56 tests pass; ArithmeticExtractor=0 vs VeriCoT=8/20, improvement_rate=0.40, honest_verdict=vericot_better; results/experiment_453_vericot_validator.json | ~21m |
| exp454 | 2026-04-18T18:07:00Z | 2026-04-18T18:10:27Z | Minimal doc updates for Exp 454 (VPRM Arithmetic Rule Verifier): appended changelog entry, appended status table row (1 experiment), verified traceability.md already has all 5 REQ-*/SCENARIO-* entries marked Implemented | ~3m27s |
| exp456 | 2026-04-18T18:42:49Z | 2026-04-18T18:56:38Z | Exp 456: Constraint Addition from Memory — ViolationPattern + ConstraintAdditionFromMemory; two-session carry-error relay; session1_fp_rate=1.0 → session2_fp_rate=0.0; fp_rate_delta=-1.0; honest_verdict=improvement; 27 tests pass; REQ-SELFLEARN-010/011/012 | ~14m |
| exp470 | 2026-04-19T00:28:31Z | 2026-04-19T00:32:01Z | Minimal doc updates for Exp 470 (PPSEBM Tier 2 Progressive Constraint Parameter Isolation): appended changelog entry, appended status table row, appended 6 new REQ-/SCENARIO- rows to traceability.md (REQ-SELFLEARN-016/017/018, SCENARIO-SELFLEARN-016/017/018); partition_isolation_score=1.0 across 3 domains | ~3m30s |
| exp473 | 2026-04-19T01:32:21Z | 2026-04-19T01:33:13Z | Minimal doc updates for Exp 473 (Milestone 2026.04.35 Retrospective): appended changelog entry only (retrospective has no new capabilities); confirmed no status/traceability updates needed; 8-criterion evaluation + 0/10 adoption finding summary | ~52s |
| minimal-docs-484 | 2026-04-19T04:50:00Z | 2026-04-19T04:52:26Z | Minimal doc updates for Exp 484 (changelog, status.md entries only) | ~2m30s | ~15k |

| planning-38 | 2026-04-19T10:28:13Z | 2026-04-19T10:28:52Z | Milestone 2026.04.38 research planning: read 10 project files + retro JSON, arxiv scan (8 papers), updated research-references.md, wrote research-roadmap-v38.md + research-roadmap-next.yaml, updated changelog + status.md | ~39s |
| exp514-docs | 2026-04-19T16:17:54Z | 2026-04-19T16:18:46Z | Minimal doc updates for Exp 514 (Live 100q Precision v7): appended changelog entry, appended status.md row, appended 4 new REQ-BENCH-014/015 + SCENARIO-BENCH-033/034 rows to traceability.md | ~52s |
| exp536-docs | 2026-04-19T22:32:48Z | 2026-04-19T22:33:18Z | Minimal doc updates for Exp 536 (Milestone 2026.04.40 Retrospective): appended changelog entry only (retrospective has no new capabilities); confirmed no status/traceability updates needed | ~30s |
| exp571-docs | 2026-04-20T09:35:32Z | 2026-04-20T09:36:23Z | Minimal doc updates for Exp 571 (HalluField Tier 0e): appended changelog entry, appended status.md row (✅ Complete), appended 3 new SCENARIO-VERIFY-154/155/156 rows to traceability.md (REQ-VERIFY-117 already existed from Exp 560) | ~51s |
| exp572-docs | 2026-04-20T09:52:45Z | 2026-04-20T09:53:54Z | Minimal doc updates for Exp 572 (PRA EBM Beam Search): appended changelog entry, appended status.md row (✅ Complete), appended 4 new REQ-REPAIR-016 + SCENARIO-REPAIR-031/032/033 rows to traceability.md | ~1m9s |
| exp579 | 2026-04-20T12:16:34Z | 2026-04-20T12:20:55Z | Exp 579 Live 50q Data Collection C: wrote scripts/experiment_579_live_data_c.py (GSM8K 200-249), tests/python/test_experiment_579_live_data_c.py (36 tests pass), added SCENARIO-DATA-016/017/018 to spec; deliverable blocked artifact written | ~4m21s |
| 2 | 2026-04-20T14:00:00Z | 2026-04-20T14:17:30Z | Doc updates for Exp 588: minimal appends to ops/changelog.md, ops/status.md (no traceability entry — retrospective has no new REQ-*/SCENARIO-*); Exp 588 logged as retrospective consolidation | ~8k |
