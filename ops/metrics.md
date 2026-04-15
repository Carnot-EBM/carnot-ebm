# Carnot — Session Metrics

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

## Session: 2026-04-15 Milestone 2026.05.06 Planning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T01:12:49Z | 2026-04-15T01:25:30Z | Plan milestone 2026.05.06 — arxiv research (6 new papers), research-references.md updated, research-roadmap-vNEXT.md (v30) + research-roadmap-next.yaml created; 13 experiments (Exps 325-337) across 4 phases | ~12m41s |

---

## Session: 2026-04-15 Exp 319 Operational Retrospective

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-15T00:32:57Z | 2026-04-15T00:43:21Z | Exp 319: Operational retrospective for milestone 2026.04.29 — 59 tests, script, artifact; 4800 pass, 99.43% coverage | ~10m24s |

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

## Session: 2026-04-14 Milestone Planning 2026.04.29

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-14T12:48:38Z | 2026-04-14T12:59:40Z | Plan milestone 2026.04.29 — read 11 project files (research-program, prd, architecture, status, changelog, research-complete, research-roadmap, change-proposals, conductor-log, research-references, hardware-wishlist), arxiv research (7 new papers via Explore agent), update research-references.md (8 new entries), create research-roadmap-vNEXT.md v29 (13 experiments across 4 phases, 3 gaps analysis, architecture diagram), create research-roadmap-next.yaml | ~40k |

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
