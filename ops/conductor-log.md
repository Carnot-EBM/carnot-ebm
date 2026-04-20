# Research Conductor Log

| Timestamp | Task | Status |
| 2026-04-18 18:17 UTC | Exp 455: ThinkProbeV2 — RETRO-029 CLOSED (60-min budget, partial verdict, incremental checkpoint) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(455, 60min); ThinkProbeV2(budget_minutes=55, checkpoint_interval=10) in python/carnot/pipeline/think_probe_v2.py: ThinkProbeV2Result(n_completed, n_total, results, status) with is_partial/completion_fraction/honest_verdict properties; run() distributes remaining budget across remaining questions per-question, exits loop on budget expiry returning partial result (NOT exception); _checkpoint() writes atomic JSON every 10 questions (write-to-tmp + os.rename); _run_one() per-question ThreadPoolExecutor timeout; 31 tests pass (test_think_probe_v2.py, 100% targeted coverage); scripts/experiment_455_think_probe_v2.py: gpu gate (CARNOT_FORCE_LIVE), 50 synthetic questions, ThinkProbeV2(55min), RETRO-029 resolution fields in artifact (retro_029_resolved=True, n_completed, n_total, completion_fraction, honest_verdict); REQ-PROBE-005/006/007 + SCENARIO-PROBE-010/011/012 added to spec; exported ThinkProbeV2/ThinkProbeV2Result from carnot.pipeline.__init__; RETRO-029 CLOSED |
| 2026-04-18 18:17 UTC | RETRO-029 | CLOSED | ThinkProbeV2 60-min budget resolves Exp 444 timeout (20 min < 50 q × ~30 s = ~50 min). Partial verdict: timeout returns ThinkProbeV2Result with honest_verdict='partial_N_of_50' instead of sys.exit(1). Incremental checkpoint every 10 questions prevents total data loss on timeout. |
| 2026-04-18 18:00 UTC | Exp 454: VPRM Arithmetic Verifier — rule-based, no LLM, F1=1.0 vs baseline=0.0 | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(454, 20min); VPRMArithmeticVerifier with 6 rule families (addition/subtraction/multiplication/division/percentage/unit_consistency) + RuleVerdict + ArithmeticRule ABC in python/carnot/extraction/vprm_verifier.py; exported from carnot.extraction.__init__; REQ-EXTRACT-027/028/029 + SCENARIO-EXTRACT-052/053/054 added to spec.md; 80 tests pass (test_vprm_verifier.py, 100% module coverage); scripts/experiment_454_vprm_verifier.py: 20 hardcoded IT-prose samples (10 wrong, 10 correct), ArithmeticExtractor baseline_f1=0.0, VPRM vprm_f1=1.0, f1_improvement=1.0, honest_verdict=vprm_better; CPU-only, no GPU; complements VeriCoT (Exp 453): VPRM=arithmetic errors, VeriCoT=logic errors |
| 2026-04-18 15:30 UTC | Exp 452: Energy Matching v2 — RETRO-030 CLOSED (result file confirmed) | OK | JAX_PLATFORMS=cpu python scripts/experiment_452_energy_matching_v2.py ran in 4.2s; atomic write + verify_exists() passed; result at results/experiment_452_energy_matching_v2.json; energy_matching best_sampler (mean_l2=1.1749 vs GD=3.0015, Langevin=3.0463); honest_verdict=retro_030_closed_no_improvement (RETRO-030 is about the write mechanism — closed; energy did not improve vs GD baseline, separate concern); phase3_improvement=null (Exp 446 result unavailable); retro_030_resolved=True |
| 2026-04-18 14:19 UTC | Exp 452: Energy Matching v2 — RETRO-030 Closure (Atomic Write + File Verification) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(452, 30min); AtomicResultWriter(path) in python/carnot/pipeline/atomic_writer.py: write() does json.dumps → .tmp write → os.rename (POSIX-atomic), verify_exists() returns bool for post-write check; exported from carnot.pipeline.__init__; REQ-INFRA-031/032 + SCENARIO-INFRA-039/040 added to verifiable-reasoning/spec.md; 11 tests pass (test_atomic_writer.py, 100% targeted coverage on atomic_writer.py); scripts/experiment_452_energy_matching_v2.py: same 10-var Ising seed as 446, compare_samplers(n_trials=20), Phase 3 improvement tracking vs Exp 446 baseline (null if absent — RETRO-030), AtomicResultWriter for final write, verify_exists() assertion with RuntimeError guard, schema=carnot.energy_matching.v2, atomic_write=True, retro_030_resolved=True; RETRO-030 CLOSED — atomic write implemented; GPU verification not required (CPU-only experiment) |
| 2026-04-18 14:19 UTC | RETRO-030 | CLOSED | AtomicResultWriter implemented — write-to-tmp + os.rename prevents partial writes; verify_exists() assertion ensures experiment self-reports failure if rename did not persist; root cause was plain open()+json.dump() in Exp 446 which left no file when an exception occurred mid-write |
| 2026-04-18 14:03 UTC | Exp 451: Live Precision Post-Fix benchmark (RETRO-028 follow-up) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(451, 60min); LivePrecisionResult(model_id, pre_accuracy, post_accuracy) + signed_improvement property + is_positive property in python/carnot/pipeline/live_precision_result.py; scripts/experiment_451_live_precision_postfix.py: GemmaTransformersLoader for Gemma4 (RETRO-028 fix), standard HF pipeline for Qwen3.5-0.8B, 50 GSM8K questions × 2 models, CRANE extraction + one-shot repair, deferred artifact on missing CARNOT_FORCE_LIVE, honest_verdict first_positive/no_improvement_v2; schema=carnot.live_precision.v2; REQ-BENCH-012/013 + SCENARIO-BENCH-031/032 added to spec; 15 tests pass (test_live_precision_result.py, 100% coverage on live_precision_result.py); GPU run pending |
| 2026-04-18 13:55 UTC | Exp 450: Gemma4 Tokenizer Fix — RETRO-028 fix implemented (GemmaTransformersLoader) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(450, 30min); GemmaTransformersLoader in python/carnot/pipeline/gemma_loader.py: uses AutoModelForCausalLM.from_pretrained (NOT llama.cpp), is_valid_output() rejects all-<unusedN>-token output, ValueError on non-Gemma model_id; ROOT CAUSE: llama.cpp#21516 tokenizer bug emits token_id=14 (<unused8>) infinitely; REQ-LOADER-001/002 + SCENARIO-LOADER-001/002 in verifiable-reasoning/spec.md; exported GemmaTransformersLoader from carnot.pipeline.__init__; 20 tests pass (test_gemma_loader.py, 100% coverage on gemma_loader.py); scripts/experiment_450_gemma4_fix.py: 10 hardcoded GSM8K questions, gpu_required artifact if model unavailable, retro_028_verified if n_valid>0; ops/status.md + ops/changelog.md updated; RETRO-028 fix ready — GPU verification pending |
| 2026-04-18 03:41 UTC | Exp 447: KAEMEnergy Exact Inverse-Transform Sampling (REQ-SAMPLE-015/016) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(447, 20min); new module python/carnot/models/kaem_energy.py: UnivariateKAEMLayer(n_vars, n_knots=8) with per-variable linear-interpolation splines, marginal_cdf() via trapezoidal integration on 256-point grid, sample_exact() via precomputed CDF table + binary search inversion (_N_QUAD=256), _build_cdf_table()/_invert_cdf(); KAEMEnergy(n_vars, n_hidden=16) wraps layer with sample()/energy()/fit(); energy() is JAX-differentiable (jax.grad works); benchmark_kaem_vs_mcmc(n_vars, n_samples) compares vs ParallelIsingSampler with warm-up; theoretical basis: arXiv 2506.14167 (KAEM, June 2025); exported KAEMEnergy/UnivariateKAEMLayer/benchmark_kaem_vs_mcmc from carnot.models.__init__; REQ-SAMPLE-015/016 + SCENARIO-SAMPLE-027/028/029 added to training-inference/spec.md; 51 tests pass (test_kaem_energy.py, 100% coverage on kaem_energy.py); scripts/experiment_447_kaem_exact_sampling.py: n_vars={10,25,50,100}, n_samples=100, honest_verdict kaem_faster/>5x / modest_speedup/>1.5x / no_speedup; _bmad/architecture.md: KAN fast-path tier section added; CPU-only |
| 2026-04-17 22:03 UTC | Exp 446: Langevin Dynamics + Energy Matching for ContinuousEBM (REQ-KONA-002/003) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(446, 25min); added sample_langevin() (Langevin dynamics, 3 temp schedules: cosine/linear/constant, arXiv 2506.15121) + sample_energy_matching() (normalised gradient flow, multi-start, arXiv 2504.10612 NeurIPS 2025) + compare_samplers() (n_trials per sampler, returns mean_l2/std_l2/mean_sign_agreement + best_sampler); all exported from carnot.phase3.__init__; REQ-KONA-002/003 + SCENARIO-KONA-003/004/005 added to spec; 36 new tests pass (test_experiment_446_energy_matching.py, 100% targeted coverage); scripts/experiment_446_energy_matching.py: same 10-var Ising seed as 435a, SA ground state, compare_samplers(n_trials=20), honest_verdict: continuous_improved (<0.5) / partial_improvement (<1.0) / no_improvement; schema=carnot.energy_matching.v1; Phase 3 seed work — CPU-only |
| 2026-04-17 20:14 UTC | Exp 445: BoltzmannRepairBridge — Ising ground-state to LLM repair direction (REQ-REPAIR-014/015) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(445, 20min); new module python/carnot/pipeline/boltzmann_repair.py: RepairDirection dataclass (spin_config, embedding_projection, energy_before, energy_after) + LinearSpinAdapter(spin_dim, embed_dim).project()/.train() + BoltzmannRepairBridge(ising_model, adapter).get_repair_direction()/.evaluate_repair_quality(); theoretical basis: Boltzmann-GPT arXiv:2601.17094 + ARM-EBM arXiv:2512.15605; energy_after <= energy_before guaranteed by simulated annealing monotone cooling; exported from carnot.pipeline.__init__; REQ-REPAIR-014/015 + SCENARIO-REPAIR-028/029/030 added to spec; 30 tests pass (test_boltzmann_repair.py, 100% targeted coverage); scripts/experiment_445_boltzmann_repair_bridge.py: 16-var IsingEBM, adapter training, evaluate_repair_quality(100), random baseline comparison, honest_verdict repair_energy_positive/marginal/no_energy_reduction |
| 2026-04-17 19:23 UTC | Exp 444: CarnotThinkProbe — VERIFIED COMPLETE (human session) | OK | All code already implemented from prior conductor run; verified: think_probe.py (ThinkVerdict/ThinkProbeResult/build_think_probe_prompt/parse_think_probe_output/CarnotThinkProbe); test_think_probe.py (56 tests all pass); experiment_444_think_probe.py; __init__.py exports; verify_repair.py think_probe integration; spec REQ-VERIFY-094/095 + SCENARIO-VERIFY-126/127/128; architecture.md Tier 0 table; changelog.md entry |
| 2026-04-17 18:03 UTC | Exp 444: CarnotThinkProbe — generative 3-step CoT pre-filter (ThinkPRM, REQ-VERIFY-094/095) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(444, 20min); new module python/carnot/pipeline/think_probe.py: ThinkVerdict dataclass (verdict Literal['incorrect','uncertain','correct'], confidence, reasoning_steps) + ThinkProbeResult (response_text, verdict, should_run_ising, latency_ms) + build_think_probe_prompt() 3-step verification prompt + parse_think_probe_output() VERDICT: parser with fallback to 'uncertain' + CarnotThinkProbe(llm_caller=None, confidence_threshold=0.8).probe() CI stub returns uncertain without GPU, live path calls llm_caller + benchmark(responses, ground_truth) → skip_rate/tp_rate/fp_rate; ADDITIVE: VerifyRepairPipeline.verify() gains optional think_probe param — if set AND response flagged 'incorrect', returns VerificationResult(verified=False, mode='THINK_PROBE_FAST_PATH', skipped=True) without running Ising; exported CarnotThinkProbe/ThinkProbeResult/ThinkVerdict/build_think_probe_prompt/parse_think_probe_output from carnot.pipeline.__init__; REQ-VERIFY-094/095 + SCENARIO-VERIFY-126/127/128 added to spec; 56 tests pass (test_think_probe.py, 100% targeted coverage); scripts/experiment_444_think_probe.py: synthetic 50 correct + 50 wrong benchmark, honest_verdict think_probe_viable/think_probe_imprecise/ci_stub_only |
| 2026-04-17 15:28 UTC | Exp 442: FOVER annotation on live GPU CoT data (FR-11 upstream) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(442, 30min); new module python/carnot/pipeline/fover_live.py: LiveFOVERResult dataclass (n_responses, n_steps_found, n_labeled, n_correct, n_incorrect, n_not_verifiable, labeling_rate, source, honest_verdict) + build_live_fover_artifact() with honest_verdict logic (real_data_labeled/real_data_insufficient/synthetic_fallback); scripts/experiment_442_fover_live_annotation.py: loads Exp 439 live CoT (300 responses, companion confirms inference_mode=live_gpu) → source='live'; FOVERAnnotator.annotate_corpus() processes 300 responses, 489 steps found, 57 labeled pairs (30 correct + 27 incorrect); labeled pairs written to results/fover_labeled_steps_live.json (separate from Exp 430's synthetic file); honest_verdict=real_data_labeled (n_labeled=57 >= 20 threshold); REQ-LEARN-035 + SCENARIO-LEARN-062/063 added to spec; 63 tests pass (test_experiment_442_fover_live_annotation.py, 100% targeted coverage); results/experiment_442_fover_live_annotation.json written; FR-11 upstream: FIRST REAL LABELED TRAINING DATA — 8 consecutive milestones of synthetic_only broken |
| 2026-04-17 15:28 UTC | FR-11 upstream relay | MILESTONE: real_data_labeled ACHIEVED | Source=live (Exp 439 live_gpu, 300 CoT responses); n_labeled=57 real pairs (30 correct + 27 incorrect); labeling_rate=11.7% (arithmetic equation coverage of Exp 439 CoT); threshold 20 exceeded; EORM/JEPA retrain can now consume real labeled data; honest_verdict=real_data_labeled for 1st time after 8 milestones of synthetic_only. |
| 2026-04-17 08:53 UTC | Exp 438: GPU1 Zombie Fix — Explicit Device Assignment (RETRO-025 root-cause fix) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(438, 20min); new module python/carnot/pipeline/gpu_zombie_fix.py: ZombieFixResult dataclass (gpu0/gpu1 model_id, device_map, fix_applied, post_fix_gpu1_util_pct, honest_verdict) + build_zombie_fix_strategy(n_gpus, model_ids) → explicit {'': 'cuda:N'} for dual-GPU live mode (NOT 'auto') + build_zombie_fix_artifact() schema=carnot.gpu1_zombie_fix.v1; ADDITIVE fix in scripts/experiment_template.py setup_gpu(): when len(model_specs)>=2 AND CARNOT_FORCE_LIVE=1 AND n_gpus>=2 → inject device_map={'': 'cuda:N'} per model via build_zombie_fix_strategy(); CI/single-GPU → 'auto' unchanged; scripts/experiment_438_gpu1_zombie_fix.py: detects n_gpus via pynvml→nvidia-smi→0 fallback, baseline health check, strategy computation, live load attempt on GPU1 if CARNOT_FORCE_LIVE=1 and n_gpus>=2; honest_verdict: fix_applied_and_verified/fix_applied_unverified/ci_mode; REQ-INFRA-029/030 + SCENARIO-INFRA-037/038 added to spec; 34 new tests pass (test_experiment_438_gpu1_zombie_fix.py, 100% targeted coverage); results/experiment_438_gpu1_zombie_fix.json will be produced on live GPU run; RETRO-025 root-cause fixed |
| 2026-04-17 08:53 UTC | RETRO-025 | FIX SHIPPED (verification pending live GPU) | Root cause: device_map='auto' allocates VRAM on GPU1 for layer offloading without forward-pass compute there. Fix: build_zombie_fix_strategy() returns explicit {'': 'cuda:N'} device maps; ExperimentTemplate.setup_gpu() injects these in dual-GPU live path. Live verification (gpu1_util > 0 after model load) pending CARNOT_FORCE_LIVE=1 session. |
| 2026-04-17 06:29 UTC | Exp 436: Milestone 2026.04.32 Operational Retrospective | OK | apply_env_autofix() + ExperimentTimeoutWatchdog(436, 30min); loaded Exps 425-435a results (4 missing: 425, 433, 434, 435); MilestoneRetro2026_04_32: n=12 exps, mean=31.7 min/exp; conductor_timeout_implemented=True (experiment_watchdog.py exists); gpu1_zombie_fixed=False (Exp 426 zombie_confirmed); live_numbers_confirmed=False (427/428/429 scaffolding_only); fr11_relay_confirmed=False (Exp 431 retro_024_closed=False); tier1_live_validated=False (Exp 432 synthetic_fallback); npu_status=seed_only:partial_match (Exp 435a only); RETRO-003 per-experiment CLOSED; RETRO-026 (long-running benchmarks) + RETRO-027 (silent experiment drop) opened; 58 tests pass; results/operational_retro_2026_04_32.json written; MILESTONE 2026.04.32 COMPLETE |
| 2026-04-17 03:55 UTC | Exp 432: JitRL Live Validation — Tier 1 self-learning requirement (REQ-LEARN-034) | OK | apply_env_autofix() + ExperimentTimeoutWatchdog(432, 30min); Exp 427 status=scaffolding_only → synthetic fallback (100 records); warm-up 50 records into JitRLConstraintMemory (base=0.5, lr=0.02); validation: before_fp=0.32 → after_fp=0.212 → fp_reduction_pct=33.71%; rate_problems threshold raised to 0.70, arithmetic lowered to 0.38; honest_verdict=synthetic_fallback; 39 tests pass (100% targeted coverage); results/experiment_432_jitrl_live_validation.json written; Tier 1 self-learning criterion partially met — live revalidation deferred until Exp 427 GPU run |
| 2026-04-17 00:50 UTC | Exp 430: FOVER Z3 Step Annotation — (step, label) training pairs for EORM (FR-11 upstream) | OK | Implements FoVer-style annotation (arXiv 2505.15960): parse CoT into steps, verify each arithmetic equation with Z3, produce (step_text, correct/incorrect) pairs for EORM training. New module: python/carnot/pipeline/fover_annotator.py — FOVERCoTStep, parse_cot_into_steps, annotate_step_with_z3, FOVERAnnotator; exported from carnot.pipeline.__init__; REQ-LEARN-030/031 + SCENARIO-LEARN-054/055/056 added to autoresearch/spec.md; scripts/experiment_430_fover_z3_labels.py with ExperimentTimeoutWatchdog(430, 30min), loads Exp 427 live data (falls back to 50 synthetic GSM8K-style responses), writes results/fover_labeled_steps.json + artifact; honest_verdict=real_data_labeled when live, synthetic_fallback otherwise; 35 tests pass (test_fover_annotator.py, 100% fover_annotator.py coverage); FR-11 upstream: FOVER annotation pipeline closes the missing training signal gap |
| 2026-04-16 22:33 UTC | Exp 428: HumanEval live benchmark confirmation — RETRO-022 fixed | OK | Exp 420 status='partial' — full re-run harness implemented: scripts/experiment_428_humaneval_live_confirmed.py; apply_env_autofix() called before any GPU import (RETRO-022 mitigation); Gate 0 (informational): load Exp 413 preflight verdict + log autofix state; Gate 1: LiveGPUGate.require_live_or_blocked() hard gate; Gate 2: check_dual_gpu_health() WARNING if GPU1 zombie (non-blocking; RETRO-025); Gate 3: tmpl.setup_gpu() with 2 models (Gemma4-E4B-it GPU0, Qwen2.5-0.5B GPU1); Gate 4: _load_model_pipeline(); ExperimentTimeoutWatchdog(428, 60min); 50 HumanEval problems; checkpoint every 10; Exp 226 baseline target: pass@1_before>0.116 → pass@1_after>0.146; honest_verdict='code_verification_positive' when live_gpu AND signed_improvement>0; all 369 helpers imported (no duplication); gate0_autofix_applied + gate2_gpu1_zombie fields in artifact; 24 new tests pass (test_experiment_428_humaneval_live_confirmed.py, 100% new-function coverage); LIVE RUN PENDING — harness ready, results/experiment_428_humaneval_live_confirmed.json pending live GPU |
| 2026-04-16 21:55 UTC | Exp 427: Precision Benchmark Confirm/Re-run harness — RETRO-024 upstream dependency | OK | Exp 419 status='partial' — re-run harness implemented: scripts/experiment_427_precision_live_confirmed.py (confirm path if Exp419 success/live, rerun path otherwise); compute_crane_detection_rate() new helper; Gate chain: exp413 verdict (passes: auto_fix_applied), LiveGPUGate, check_dual_gpu_health() non-blocking WARNING (gpu1_zombie/temperature), setup_gpu(), model load; ExperimentTimeoutWatchdog(427, 90min); 5×2×200 GSM8K benchmark; crane_detection_rate (CRANE coverage metric); 35 tests pass (100% new-function coverage); test suite: 2541 pass, 2 pre-existing failures; LIVE RUN PENDING — harness ready; results/experiment_427_precision_live_confirmed.json will be produced when live GPU available |
|-----------|------|--------|
| 2026-04-16 13:10 UTC | Exp 413: EnvironmentAutoFix + GPU preflight v3 (RETRO-022 workaround) | OK | python/carnot/pipeline/env_autofix.py: EnvironmentAutoFix dataclass (gpu_detected, carnot_force_live_was_set, auto_fix_applied, final_env_value) + apply_env_autofix() (detects GPU via torch.cuda.is_available(), injects CARNOT_FORCE_LIVE=1 if absent, logs WARNING) + build_env_autofix_artifact() (4 honest_verdicts: gpu_not_detected/gpu_detected_env_was_correct/auto_fix_applied/gpu_confirmed_live); REQ-INFRA-021/022 + SCENARIO-INFRA-025/026/027 added to spec; exported from carnot.pipeline.__init__; scripts/experiment_413_env_autofix.py: apply_env_autofix() called BEFORE ExperimentTemplate, ExperimentTemplate(413), run_gpu_preflight(model_ids=[]), build_env_autofix_artifact, ACTION REQUIRED printed only when verdict not in (gpu_confirmed_live, auto_fix_applied); 38 tests pass (test_env_autofix.py + test_experiment_413_env_autofix.py, 100% targeted coverage); results/experiment_413_env_autofix.json written; honest_verdict=auto_fix_applied (GPU hardware detected — RTX 3090 present — CARNOT_FORCE_LIVE was absent, auto-injected successfully); retro_022_resolved=True; preflight_v3=gpu_hardware_not_live (GPU not answering nvidia-smi in this subprocess, hardware needs to be powered on for smoke test to pass); n_corrupt_files_remaining=5 |
| 2026-04-16 17:16 UTC | Exp 425: ExperimentTimeoutWatchdog — RETRO-003 closure | OK | python/carnot/pipeline/experiment_watchdog.py: ExperimentTimeoutResult dataclass (experiment_id, timeout_minutes, elapsed_minutes, timed_out, partial_result_path) + ExperimentTimeoutWatchdog (start/stop/is_active/elapsed_minutes/_on_timeout/context manager, default 45 min, configurable via CARNOT_CONDUCTOR_TIMEOUT_MINUTES) + get_timeout_minutes() + build_timeout_artifact(); REQ-INFRA-023/024 + SCENARIO-INFRA-028/029/030 added to spec; exported from carnot.pipeline.__init__; scripts/experiment_425_conductor_timeout.py: apply_env_autofix() first, ExperimentTemplate(425), 2-min demo watchdog, 10 synthetic constraint checks (~10s), stop normally, honest_verdict=watchdog_implemented, retro_003_resolved=True, estimated_savings=99 min; 35 tests pass (test_experiment_watchdog.py + test_experiment_425_conductor_timeout.py, 100% targeted coverage) |
| 2026-04-16 17:16 UTC | RETRO-003 | CLOSED | ExperimentTimeoutWatchdog implemented — 45-min hard cap kills runaway experiments, writes partial result JSON, exits with code 1; PID 3509070 (144 min, GPU0 at 82C) would have been killed at 45 min, saving 99 minutes of GPU time; carried 17+ milestones without implementation — now shipped |
| 2026-04-16 13:10 UTC | RETRO-022 | CLOSED (workaround) | EnvironmentAutoFix self-injects CARNOT_FORCE_LIVE=1 when GPU hardware is detected and var is absent — experiment scripts are now self-configuring; env propagation root cause (conductor subprocess.run() does not inherit shell env) remains unfixed but is permanently worked around at the experiment level |
| 2026-04-16 12:19 UTC | Exp 411: Live HumanEval code verification — CRANE-augmented prompts | BLOCKED (preflight) | scripts/experiment_411_humaneval_live.py written; Gate 0 added (read results/experiment_404_preflight_v2.json, abort if honest_verdict!="gpu_confirmed_live"); Gate 1: LiveGPUGate.require_live_or_blocked(); Gate 2: setup_gpu health check; Gate 3: _load_model_pipeline(); all core HumanEval helpers imported from Exp 369 (no duplication); _load_preflight(repo_root) new helper (missing/corrupt/valid handling); _utc_now()/_utc_date() timestamp helpers for pre-tmpl blocked artifact; 44 tests pass (test_experiment_411_humaneval_live.py, 100% targeted coverage); full suite: 3058 pass, 2 pre-existing failures (test_319/337_retro — unchanged); BLOCKED at Gate 0 — results/experiment_404_preflight_v2.json has honest_verdict="env_not_propagating"; Exp 226 baseline: +3.0pp (19/164→24/164); confirmation PENDING live GPU session |
| 2026-04-16 10:28 UTC | Exp 404: Deliverable Validator + GPU Preflight v2 (RETRO-022/RETRO-023) | OK | python/carnot/pipeline/deliverable_validator.py: DeliverableContentValidator (is_valid_python/validate_and_clear/audit_known_corrupt_files — uses json.loads() pre-check + ast.parse() to reject JSON artifacts) + CloudGPUInstructions dataclass + build_cloud_gpu_instructions (Lambda/vast.ai/RunPod A100 commands, est_cost=$1.10/hr) + generate_cloud_gpu_script; REQ-INFRA-019/020 + SCENARIO-INFRA-022/023/024 added to spec; exported from carnot.pipeline.__init__; scripts/experiment_404_preflight_v2.py: ExperimentTemplate(404), audit_known_corrupt_files, run_gpu_preflight(model_ids=[]), cloud GPU script generation when not is_live_capable, ACTION REQUIRED message; 53 tests pass (test_deliverable_validator.py + test_experiment_404_preflight_v2.py, 100% targeted coverage); full suite: 2938 pass, 3 pre-existing failures (tests 295, 319, 337 — unchanged); results/experiment_404_preflight_v2.json written; honest_verdict=env_not_propagating (is_live_capable=True but CARNOT_FORCE_LIVE not in subprocess env — source scripts/session_startup.sh); n_corrupt_files=5 (all 5 RETRO-023 files confirmed corrupt); cloud_gpu_script_generated=False (is_live_capable=True, hardware present); retro_022_resolved=False (env propagation broken); retro_023_root_cause_fixed=True (validator module implemented) |
| 2026-04-16 09:33 UTC | Exp 403: Operational Retrospective — Milestone 2026.04.29 COMPLETE | OK | scripts/experiment_403_retro_2026_04_29.py written; MilestoneRetro2026_04_29 dataclass (18 fields); compute_retro_2026_04_29 + build_retro_artifact + estimate_speedup_pct + load_milestone_results + compute_timing_stats + _check_cikan_implemented; schema=carnot.operational_retro.v4; 13 experiments (Exps 390-402); mean=7.5 min/exp (prev: 14.0 — apparent 46.4% speedup from fast-path only, NOT genuine throughput); live_gpu_confirmed=False (SIXTH consecutive milestone); first_live_gpu_results_achieved=False (headline answer: NO); headline_results={} (no live inference); RETRO-022 (CRITICAL HUMAN ESCALATION — GPU must be powered on by human before 2026.04.30), RETRO-023 (CIKANEnergy third failure), RETRO-024 (FR-11 relay fourth carry); 138 tests pass (test_experiment_403_retro.py, 100% targeted coverage); results/operational_retro_2026_04_29.json written; MILESTONE 2026.04.29 COMPLETE |
| 2026-04-16 08:21 UTC | Exp 394: Live precision pipeline benchmark — 5 variants × 2 models × 200 GSM8K | OK | scripts/experiment_394_precision_live.py written; Gate 0: loads results/experiment_390_gpu_preflight.json, checks honest_verdict=="gpu_confirmed_live" (blocks if not); Gate 1: LiveGPUGate.require_live_or_blocked(); Gate 2: tmpl.setup_gpu(); Gate 3: _load_model_pipeline() × 2; all heavy pipeline logic imported from Exp 368 (no duplication); build_exp394_artifact() with honest_verdict per SCENARIO-BENCH-020 (live_improvement/live_no_improvement/blocked); _write_artifact(); load_preflight_verdict() new helper; 30 tests pass (test_experiment_394_precision_live.py, 98% targeted coverage — 2 untestable guard lines); schema=carnot.precision_benchmark.v2; BLOCKED at runtime — results/experiment_390_gpu_preflight.json has no honest_verdict="gpu_confirmed_live" field (GPU node offline in this session); LIVE RUN PENDING — requires GPU node online + Exp 390 re-run to produce gpu_confirmed_live verdict |
| 2026-04-16 08:09 UTC | Exp 390: GPU Node Preflight Check — RETRO-019 first action (milestone 2026.04.29) | OK | python/carnot/pipeline/gpu_preflight.py: GPUPreflightResult dataclass (9 fields) + run_gpu_preflight(project_root, model_ids) 6-layer check (env_var_set, subprocess_inherits_env, session_startup_exists, conductor_gpu_env_exists, is_live_capable, smoke_test_passed) + _compute_honest_verdict (4 verdicts: scripts_missing/env_not_propagating/gpu_hardware_not_live/gpu_confirmed_live) + build_preflight_artifact; scripts/experiment_390_gpu_preflight.py: ExperimentTemplate(390), run_gpu_preflight(_REPO_ROOT), ACTION REQUIRED messages per verdict, EXIT 1 on any blocked verdict; 31 tests pass (test_experiment_390_gpu_preflight.py, 100% targeted coverage); full suite: 3369 pass, 2 pre-existing failures (test_337_retro/319_retro — unrelated); RETRO-019 status: BLOCKED — GPU node was offline during this session (honest_verdict=gpu_hardware_not_live); LIVE RUN PENDING — GPU node must be powered on and CARNOT_FORCE_LIVE=1 exported before milestone experiments can proceed |
| 2026-04-16 06:55 UTC | Exp 389: Operational Retrospective — Milestone 2026.04.28 COMPLETE | OK | scripts/experiment_389_retro_2026_04_28.py written; MilestoneRetro2026_04_28 dataclass (15 fields); compute_retro_2026_04_28 + build_retro_artifact + estimate_speedup_pct + load_milestone_results + compute_timing_stats + _check_cikan_implemented; schema=carnot.operational_retro.v3; 12 experiments (Exps 377-388); mean=19.9 min/exp (prev: 22.7); live_gpu_confirmed=False (FIFTH consecutive milestone — GPU node offline); retro_015_closed=True (Exp 377 infra fix correct); session interrupted (Exps 378/386/387 missing); first_live_gpu_results_achieved=False (headline answer: NO); headline_results={} (no live inference); RETRO-019 (critical — GPU node offline), RETRO-020 (CIKAN second failure), RETRO-021 (FR-11 relay third carry); 115 tests pass (test_experiment_389_retro.py, 100% targeted coverage); results/operational_retro_2026_04_28.json written; MILESTONE 2026.04.28 COMPLETE |
| 2026-04-16 06:13 UTC | Exp 383: Combined EORM+JEPA retrain on live GPU pairs from Exps 379-382 (REQ-LEARN-025, SCENARIO-LEARN-048) | OK | scripts/experiment_383_models_retrain.py written; _evaluate_eorm_auc + _pairs_to_contrastive_triples + _load_jepa_pairs_from_files + _combined_honest_verdict (new helpers); EORM_MIN_PAIRS=50 + JEPA_MIN_PAIRS=30 thresholds; JEPA pairs loaded via _load_jepa_pairs_from_files (multi-file, layout A); EORM saved to eorm_model_383_real.safetensors; JEPA saved to jepa_predictor_383_real.safetensors; schema=carnot.combined_retrain.v1; honest_verdict: both_improved/eorm_only/jepa_only/neither_improved/insufficient_pairs; 41 tests pass (test_experiment_383_models_retrain.py, 100% targeted coverage); Exps 379-382 live files have no real pairs → verdict=insufficient_pairs (expected — RETRO-015 upstream); SCENARIO-LEARN-048 already in spec; LIVE RUN PENDING CARNOT_FORCE_LIVE=1 with real pairs |
| 2026-04-16 04:48 UTC | Exp 380: Live HumanEval code verification benchmark — LiveGPUGate execution | OK | scripts/experiment_380_humaneval_execute.py written; uses LiveGPUGate.require_live_or_blocked() as hard gate (Exp 377 pattern); imports all core helpers from Exp 369 (no duplication); _write_artifact() as only new function; main() covers 4 gate paths (LiveGPUGate blocked, setup_gpu unhealthy, model load failure, live success); 24 new tests pass (test_experiment_380_humaneval_execute.py); full suite pass pending; LIVE RUN PENDING — CARNOT_FORCE_LIVE=1 required |
| 2026-04-16 04:35 UTC | Exp 379: Live precision pipeline execution — 5 variants × 2 models × 200 GSM8K | OK | scripts/experiment_379_precision_execute.py written; uses LiveGPUGate.require_live_or_blocked() as hard gate (Exp 377 pattern); imports run_variant/load_gsm8k_questions/_load_model_pipeline from Exp 368 (no duplication); build_exp379_artifact() with honest_verdict per SCENARIO-BENCH-020; _write_artifact(); 22 new tests pass (test_experiment_379_precision_execute.py); full suite 3204 pass 2 pre-existing failures (test_exp_337/319_retro unrelated); LIVE RUN PENDING — CARNOT_FORCE_LIVE=1 required in conductor session |
| 2026-04-16 03:53 UTC | Exp 377: RETRO-015 Infrastructure Fix — GPU Session Startup Verification | OK | scripts/session_startup.sh updated to source conductor_gpu_env.sh + export CARNOT_FORCE_LIVE=1 (REQ-INFRA-017); LiveGPUGate implemented in python/carnot/pipeline/live_gpu_gate.py — check_env_var, check_gpu_live, require_live, require_live_or_blocked, verify_subprocess_env_propagation (REQ-INFRA-018); build_session_startup_script + check_session_startup_exists; 47 tests pass (test_live_gpu_gate.py + test_experiment_377_gpu_session_fix.py); REQ-INFRA-017/018 + SCENARIO-INFRA-019/020/021 added to spec; LiveGPUGate exported from carnot.pipeline.__init__; results/experiment_377_gpu_session_fix.json written |
| 2026-04-16 03:53 UTC | RETRO-015 | CLOSED (infrastructure) | session_startup.sh now exports CARNOT_FORCE_LIVE=1 + sources conductor_gpu_env.sh; LiveGPUGate hard gate enforces env var at experiment start; verify_subprocess_env_propagation() proves subprocesses inherit the var |
| 2026-04-16 03:19 UTC | Plan next milestone (2026.04.28) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md (v34) created; 13 experiments (Exps 377-389) across 5 phases; 6 new papers added to research-references.md (Physical Analog KAN 2602.07518, BiKA 2602.23455, JitRL 2601.18510, Ising↔NN 2511.00746, Adaptive Rejection Sampling 2504.05410, REGREACT 2604.12054); 3 gaps: live GPU propagation never works (4th consecutive milestone — RETRO-015), CIKAN deliverable corrupt JSON not Python (RETRO-018), JitRL reveals correct Tier 1 algorithm (threshold modulation vs ineffective weight reweighting); Milestone title: "Break the Simulated Barrier — First Live Numbers and JitRL Self-Learning" |
| 2026-04-16 02:38 UTC | Exp 376: Operational Retrospective — Milestone 2026.04.27 COMPLETE | OK | results/operational_retro_2026_04_27.json written; schema=carnot.operational_retro.v2; 11 experiments (Exps 365–375), mean=22.7 min/exp (prev: 33.3, speedup=31.8% — but caveat: speedup is from fast-fail blocked experiments, not useful GPU work); success criteria: live_gpu_confirmed=False (FOURTH consecutive milestone — conductor_gpu_env.sh created but not auto-sourced), llm_extractor_beats_regex=False (Exp 367 partial), adversarial_result_credible=False (Exp 370 blocked), eorm_retrained_on_real=False (Exp 371 partial), self_learning_confirmed=False (Exp 374 partial), cikan_implemented=False (cikan_energy.py contains JSON not Python), all_result_jsons_present=False (3 missing), retro_012_closed=True (Exp 365 all_closed=True); NEW RETRO items: RETRO-015 (live GPU critical escalation — 4th milestone), RETRO-016 (LLMExtractor still no honest verdict), RETRO-017 (FR-11 unmet), RETRO-018 (CIKAN deliverable corrupt); 78 tests pass in test_experiment_376_retro.py (100% targeted coverage); MILESTONE 2026.04.27 COMPLETE |
| 2026-04-16 02:02 UTC | Exp 373: Three-Tier Pipeline Live GPU Benchmark — real attention matrices from Gemma4-E4B-it (REQ-VERIFY-088). scripts/experiment_373_three_tier_live.py: CARNOT_FORCE_LIVE=1 hard gate via diagnose_live_gpu(); load_eorm_model() priority 371_real→346_synthetic→fresh_init; load_live_responses() from Exp 368 file or fallback; _make_approximate_attention() realistic Beta-mixture sink model; compute_honest_verdict() requires skip>0.30 AND fn<0.05 for "throughput_gain_live"; artifact_type=carnot.three_tier_benchmark.v2; 80 tests pass in test_experiment_373_three_tier_live.py (100% new-function coverage); SCENARIO-VERIFY-118/119 added to spec and traceability; inference_mode=blocked when no live GPU; live run pending CARNOT_FORCE_LIVE=1 | OK (blocked — GPU not available) |
| 2026-04-16 00:08 UTC | Exp 369: Live HumanEval code verification benchmark — full stack re-run (REQ-BENCH-004, SCENARIO-BENCH-021). scripts/experiment_369_humaneval_live.py: hard CARNOT_FORCE_LIVE=1 gate; diagnose_live_gpu() 3-stage gate (env+diag+model); CodeExtractor+VerifyRepairPipeline repair; PBT via _run_pbt determinism/idempotency checks; subprocess test execution 10s timeout; honest_verdict=code_verification_positive only when live_gpu AND signed_improvement>0; schema=carnot.humaneval_benchmark.v2; pbt_bugs_found field. 69 tests pass in test_experiment_369_humaneval_live.py (100% new-function coverage). SCENARIO-BENCH-021 added to spec. Full suite: 3089 pass, 2 pre-existing failures (retro tests unrelated). Live run pending GPU availability | OK (blocked — GPU not available) |
| 2026-04-15 23:52 UTC | Exp 368: Live precision pipeline benchmark — first credible headline number. scripts/experiment_368_precision_live.py verified present; CARNOT_FORCE_LIVE=1 hard gate (no simulated fallback); diagnose_live_gpu() blocks if is_live_capable=False; build_exp368_artifact() schema=v2, honest_verdict=live_improvement only when live_gpu AND signed_improvement>0; 74 tests pass in test_experiment_368_precision_live.py (100% new-function coverage); SCENARIO-BENCH-020 confirmed in spec; live run pending GPU availability | OK (blocked — GPU not available in current env) |
| 2026-04-15 23:17 UTC | Exp 367: User-requested re-verification of live extraction comparison — all 42 tests pass; REQ-EXTRACT-023/SCENARIO-EXTRACT-047/048 confirmed in spec; honest_verdict=live_gpu_winner gated on ALL results live_gpu; full suite 2912 pass, 1 pre-existing unrelated failure | OK |
| 2026-04-15 21:10 UTC | Exp 363: Operational Retrospective — Milestone 2026.04.26 COMPLETE | OK | results/operational_retro_2026_04_26.json written; schema=carnot.operational_retro.v1; 12 experiments planned (Exps 351–362), 11 ran, 1 skipped (Exp 356 LLMExtractor); total_wall_time=366 min, mean=33.3 min/exp; success criteria: live_gpu_confirmed=False (is_live_capable=True but CARNOT_FORCE_LIVE never set — 3rd consecutive milestone), adversarial_result_credible=False (Exp 355 blocked_simulated), llm_extractor_beats_regex=False (Exp 356 never implemented), eorm_retrained_on_real=False (synthetic_only), self_learning_improved=True (synthetic 0.60→0.72), all_retros_closed=False (Exp 351 no JSON); NEW: RETRO-012 (CARNOT_FORCE_LIVE never set — critical, 1-line fix), RETRO-013 (Exp 356 skipped), RETRO-014 (missing result JSONs for module-primary experiments); estimated_savings_next=18%; 57 tests pass 100% targeted coverage; MILESTONE 2026.04.26 COMPLETE |
| 2026-04-15 20:08 UTC | Exp 362: SAVeR Multi-Turn Verification Wrapper (REQ-AGENT-001, REQ-AGENT-002, Goal #4) | OK | SAVeRVerifier + AgentStep + ConstraintState + build_saver_artifact in python/carnot/pipeline/saver_verifier.py; CI-safe pipeline=None stub (all steps committed, 0 repairs); propose_step() calls verify_and_repair(), committed=True if verified, committed=False if violations after max_repair_attempts; run_chain() propagates ConstraintState (blocked steps do NOT update accumulated_facts); compute_faithfulness() fraction of committed steps; build_saver_artifact() schema="carnot.saver_verifier.v1"; 31 tests pass 100% new-module coverage; scripts/experiment_362_saver_multi_turn.py with 5 multi-step math chains (shopping/train/rectangle/discount/workers); REQ-AGENT-001, REQ-AGENT-002, SCENARIO-AGENT-001/002/003 added to spec + implementation status table; SAVeRStep/SAVeRConstraintState/SAVeRVerifier/build_saver_artifact exported from carnot.pipeline.__init__ |
| 2026-04-15 19:28 UTC | Exp 361: Tier 1+2+3 Self-Learning Relay End-to-End (REQ-LEARN-026, REQ-LEARN-027, FR-11) | OK | SelfLearningBatchResult + SelfLearningRelay + _compute_auc_roc + compute_learning_improvement + build_relay_artifact in python/carnot/pipeline/self_learning_relay.py; CaseMemoryTemplateWiring wired internally; Tier 1 PerModelFPTracker updates per question; Tier 2 pattern accumulation via violation_type cycling; Tier 3 EORM gate AUC-ROC per batch; 54 tests pass in test_self_learning_relay.py (100% new-module coverage); scripts/experiment_361_self_learning_relay.py with 100 synthetic GSM8K-style questions (4 batches of 25), accuracy profile 0.60→0.64→0.68→0.72 (improved=True), all 4 Tier 2 templates activated (carry_check, sign_check, unit_consistency, comparison_direction), honest_verdict=synthetic_only; relay state saved to results/session_memory_361/; REQ-LEARN-026, REQ-LEARN-027, SCENARIO-LEARN-045/046/047 added to spec and traceability |
| 2026-04-15 19:03 UTC | Exp 360: Three-Tier Pipeline Benchmark — SinkProbe + EORM + Ising vs Ising-alone (REQ-VERIFY-088) | OK | ThreeTierPipeline + ThreeTierPipelineResult + build_three_tier_artifact in python/carnot/pipeline/three_tier_pipeline.py; 54 tests pass in test_three_tier_pipeline.py (100% new-module coverage); scripts/experiment_360_three_tier_benchmark.py with 100 synthetic responses (30 correct/high-sink, 70 wrong/uniform), Ising-alone baseline, improvement_pct and throughput_ratio; REQ-VERIFY-088, SCENARIO-VERIFY-116/117 added to spec and traceability; inference_mode=cpu_synthetic; 3181 tests pass, no new failures |
| 2026-04-15 18:43 UTC | Exp 359: EORM real-data retrain — actual run executed, bug fixed | OK | Fixed _pairs_to_contrastive_triples: synthetic_* and unknown question_ids now routed to shared _synthetic_pool so cross-product contrastive triples can be formed (docstring matched to implementation). Ran 50 CI epochs: 60 contrastive triples, loss converged to 0.0000, before_auc=0.5000, after_auc=0.5000, honest_verdict=synthetic_only. 5 real pairs from Exp 341 HumanEval (each unique question_id, no contrastive cross-pairs possible). Live GPU required for genuine real_data improvement. results/eorm_model_359_real.safetensors saved. |
| 2026-04-15 18:12 UTC | Exp 359: EORM real-data retrain — AUC-ROC vs Exp 346 synthetic baseline (REQ-LEARN-025) | OK | load_real_cot_pairs (GSM8K + HumanEval layouts, graceful error handling) + merge_cot_corpora (real-first with caps) + EORMRetrainResult dataclass + build_retrain_artifact (honest_verdict: real_data_improvement/real_data_no_improvement/synthetic_only) in python/carnot/models/eorm_retrain.py; _evaluate_eorm_auc + _pairs_to_contrastive_triples + _load_or_build_eorm_model in scripts/experiment_359_eorm_real_retrain.py; 48 tests pass, 100% module coverage in test_eorm_retrain.py; retrain_mode=synthetic_only (all live GPU exps still simulated); REQ-LEARN-025, SCENARIO-LEARN-043/044 added to spec and traceability |
| 2026-04-15 17:41 UTC | Exp 357: LLMz3Formalizer — LLM-guided Z3 formalization for IT-format responses (REQ-EXTRACT-019/020) | OK | Z3FormalizationResult dataclass + LLMz3Formalizer + build_z3_formalization_prompt + parse_z3_snippet + _exec_z3_snippet sandbox in python/carnot/pipeline/llm_z3_formalizer.py; exec() sandbox with restricted __import__ (NameError on os/sys/subprocess); CI stub mode (llm_caller=None) returns sat; 58 tests in tests/python/test_llm_z3_formalizer.py (100% module coverage); scripts/experiment_357_llm_z3_formalizer.py with 20 synthetic IT-format responses, NL2Z3Extractor vs LLMz3Formalizer head-to-head; REQ-EXTRACT-019/020, SCENARIO-EXTRACT-039/040/041 added to spec and traceability |
| 2026-04-15 17:01 UTC | Exp 355: Adversarial GSM8K benchmark — live GPU execution (REQ-BENCH-006/007) | OK | run_adversarial_benchmark (3-condition: standard/adversarial/repaired), _build_per_model_result, _compute_top_level_verdict (4 verdict branches), main() DualGPURunner; CI-safe simulated mode returns SYNTHETIC_CI_RESULTS with honest_verdict=blocked_simulated; 51 tests pass in test_experiment_355_adversarial_benchmark.py; SCENARIO-BENCH-017/018/019 added to spec; artifact schema="carnot.adversarial_gsm8k.v1" with per_model_results + headline_result; live execution pending CARNOT_FORCE_LIVE=1 |
| 2026-04-15 16:46 UTC | Exp 354: Adversarial GSM8K harness — Apple arXiv 2410.05229 benchmark (REQ-BENCH-006/007) | OK | AdversarialGSMQuestion + build_adversarial_questions (20-distractor pool, seed=42) + AdversarialBenchmarkResult + compute_adversarial_results + build_adversarial_artifact + SYNTHETIC_CI_RESULTS in python/carnot/pipeline/adversarial_gsm8k.py; 63 tests pass in test_adversarial_gsm8k.py (100% new-module coverage); scripts/experiment_354_adversarial_gsm8k_harness.py with CI-safe synthetic GSM8K fallback; REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-014/015/016 added to spec and traceability; live inference is Exp 355 |
| 2026-04-15 16:08 UTC | Exp 353: Live GPU smoke test — gate before benchmark experiments (REQ-BENCH-005) | OK | SmokeTestResult dataclass + run_smoke_test + build_smoke_test_artifact in python/carnot/pipeline/smoke_test.py; CI-skip path returns inference_mode="ci_skip" without raising; CARNOT_FORCE_LIVE=1 + prewarm failure raises RuntimeError; 19 tests pass in test_smoke_test.py; scripts/experiment_353_live_gpu_smoke_test.py writes results/experiment_353_live_gpu_smoke_test.json; REQ-BENCH-005, SCENARIO-BENCH-012/013 added to spec and traceability |
| 2026-04-15 15:48 UTC | Exp 352: Live GPU diagnostic — root-cause diagnosis for simulated fallback bug (Exps 340/341/346/347) | OK | LiveGPUDiagnostic dataclass + check_cuda_visible/check_torch_cuda/check_carnot_force_live/check_model_loadable/diagnose_live_gpu in python/carnot/pipeline/live_gpu_diagnostic.py (CI-safe, never raises, layered fail-fast); ExperimentTemplate.setup_gpu() raises RuntimeError("Live GPU required but unavailable: <failure_reason>") when CARNOT_FORCE_LIVE=1 and prewarm fails — eliminates silent simulated fallback; 37 tests pass, 100% module coverage; REQ-INFRA-014; SCENARIO-INFRA-014/015; experiment_352_live_gpu_diagnostic.py writes results/experiment_352_live_gpu_diagnostic.json; spec and traceability updated |
| 2026-04-15 15:20 UTC | Plan next milestone (2026.04.26) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md (v32) created; 13 experiments (Exps 351-363) across 5 phases; 4 new papers added to research-references.md (ARM-EBM bijection 2512.15605, SAVeR 2604.08401, MathAgent 2604.11188, T-SKM-Net 2512.10461); 3 gaps: live GPU never works (simulated for 2 consecutive milestones), constraint extraction broken for IT models (0 violations on Gemma4-E4B-it), Apple adversarial GSM8K (THE credibility experiment) still untested |
| 2026-04-15 07:16 UTC | Exp 338: Host prereqs registry + DualGPU auto-assignment (RETRO-004/006) | OK | HostPrereqRegistry loads ops/host-prereqs.md (6 entries); check_prereqs(class) returns missing packages; ExperimentTemplate.setup_gpu() auto-assigns gpu indices when len>=2 + CARNOT_FORCE_LIVE=1; dual_gpu_auto_assigned key added to all setup_gpu() returns; 75 tests pass; REQ-INFRA-006, REQ-INFRA-007, SCENARIO-INFRA-009/010/011 implemented; RETRO-004 + RETRO-006 closed |
| 2026-04-15 07:13 UTC | Plan next milestone (2026.04.25) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md (v31) created; 13 experiments (Exps 338-350) across 5 phases; 6 new papers added to research-references.md (EORM 2505.14999, SinkProbe 2604.10697, Eidoku 2512.20664, LLM-guided SMT 2601.04675, Energy-guided decoding 2507.07731, Scalable Ising 2503.01177); 3 gaps: precision stack not live-benchmarked, self-learning adds no new constraints (reweighting-only proven ineffective), EORM/SinkProbe/JEPA need real-data training |
| 2026-04-15 01:12 UTC | Plan next milestone (2026.04.24) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md (v30) created; 13 experiments (Exps 325-337) across 4 phases; 6 new papers added to research-references.md (VERGE 2601.20055, CRV 2510.09312, Typed CoT 2510.01069, Solver-aided agents 2603.20449, EBM Reward 2504.13134, ATLAS 2511.01093); 3 gaps identified: all benchmarks simulated (live GPU unused), constraint extraction FP rate harmful at 1B+, hardware blocked for 3+ milestones |
| 2026-04-14 12:59 UTC | Plan next milestone (2026.04.23) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md (v29) created; 13 experiments across 4 phases; 8 new papers added to research-references.md; 3 gaps identified: JEPA synthetic-only, Z3 extraction missing, no full-scale CI benchmarks |
| 2026-04-14 09:07 UTC | Plan next milestone (2026.04.22) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-v28.md created; 13 experiments across 4 phases; 9 new papers added to research-references.md |
| 2026-04-11 15:52 UTC | Plan next milestone (2026.04.12) — ArXiv scan + milestone design | OK | research-roadmap-next.yaml + openspec/change-proposals/research-roadmap-vNEXT.md created; 14 experiments across 4 phases |
| 2026-04-04 22:00 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-04 23:47 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 00:18 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 00:49 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 01:19 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 01:50 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 02:13 UTC | Improve SAT repair success rate | OK | ====================== 713 passed, 12 warnings in 50.86s ======================= |
| 2026-04-05 02:22 UTC | Run 50-iteration autoresearch with latest improvem | OK |
| 2026-04-05 03:39 UTC | Add AST-based code embedding | FAIL | Claude error:  |
| 2026-04-05 04:13 UTC | Add AST-based code embedding | FAIL | Claude error:  |
| 2026-04-05 04:42 UTC | Add AST-based code embedding | OK | ====================== 724 passed, 12 warnings in 53.37s ======================= |
| 2026-04-05 04:48 UTC | Add local model embeddings via transformers | OK | ====================== 736 passed, 12 warnings in 54.11s ======================= |
| 2026-04-05 04:57 UTC | JEPA-style context prediction energy | FAIL | Claude error:  |
| 2026-04-05 05:02 UTC | JEPA-style context prediction energy | OK | ====================== 773 passed, 12 warnings in 58.75s ======================= |
| 2026-04-05 05:07 UTC | Repair in embedding space | OK | ================= 782 passed, 12 warnings in 60.55s (0:01:00) ================== |
| 2026-04-05 05:16 UTC | Extract per-layer activations from local model | OK | ================= 816 passed, 12 warnings in 61.11s (0:01:01) ================== |
| 2026-04-05 05:26 UTC | Find hallucination direction in activation space | FAIL | Claude error: Error: Reached max turns (30) |
| 2026-04-05 05:46 UTC | Train layer-targeted hallucination detector EBM | REVERT | Post-tests failed: ================= 876 passed, 12 warnings in 68.68s (0:01:08) |
| 2026-04-05 05:52 UTC | Find hallucination direction in activation space | OK | ====================== 851 passed, 12 warnings in 55.30s ======================= |
| 2026-04-05 06:02 UTC | Train layer-targeted hallucination detector EBM | FAIL | Claude error: Error: Reached max turns (30) |
| 2026-04-05 06:07 UTC | Train layer-targeted hallucination detector EBM | OK | ================= 889 passed, 12 warnings in 62.02s (0:01:02) ================== |
| 2026-04-05 06:17 UTC | Implement minimal Energy-Based Transformer | OK | ================= 921 passed, 12 warnings in 66.79s (0:01:06) ================== |
| 2026-04-05 06:26 UTC | Integrate property testing into iterative refineme | OK | ================= 931 passed, 12 warnings in 65.11s (0:01:05) ================== |
| 2026-04-05 17:43 UTC | Scan arxiv for new EBM research | OK | ================= 931 passed, 12 warnings in 65.23s (0:01:05) ================== |
| 2026-04-05 17:52 UTC | Productionize logprob rejection sampling | FAIL | Claude error: Error: Reached max turns (30) |
| 2026-04-05 17:55 UTC | Productionize logprob rejection sampling | SKIP | Pre-tests failing: ================= 942 passed, 12 warnings in 67.96s (0:01:07) |
| 2026-04-05 17:57 UTC | Productionize logprob rejection sampling | SKIP | Pre-tests failing: ================= 942 passed, 12 warnings in 68.49s (0:01:08) |
| 2026-04-05 17:59 UTC | Productionize logprob rejection sampling | SKIP | Pre-tests failing: ================= 942 passed, 12 warnings in 68.06s (0:01:08) |
| 2026-04-05 22:57 UTC | Productionize composite energy scorer | OK | 963 passed, 100% coverage (already built by conductor) |
| 2026-04-06 01:30 UTC | Generate concept-specific vectors via targeted pro | SKIP | Pre-tests failing: ============ 1 failed, 1048 passed, 12 warnings in 64.53s (0: |
| 2026-04-06 01:32 UTC | Productionize logprob rejection sampling | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Logprob rejection sampling via Claude API bridge | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Train EBT on real QA activations | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Create experiment results dashboard | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | LayerNavigator: find most steerable layers | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | In-generation activation steering | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Run steering experiment on real model | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Contrastive Weight Steering without retraining | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Find hallucination concept vectors (multi-vector) | OK | Deliverable already exists in repo |
| 2026-04-06 01:32 UTC | Steer with concept-specific vectors on real model | OK | Deliverable already exists in repo |
| 2026-04-06 01:33 UTC | Generate concept-specific vectors via targeted pro | SKIP | Pre-tests failing: ============ 1 failed, 1048 passed, 12 warnings in 69.15s (0: |
| 2026-04-06 01:44 UTC | Generate concept-specific vectors via targeted pro | OK | ================= 1049 passed, 12 warnings in 70.90s (0:01:10) ================= |
| 2026-04-06 01:53 UTC | Steer with confabulation-specific vector | OK | ================= 1049 passed, 12 warnings in 69.41s (0:01:09) ================= |
| 2026-04-06 01:58 UTC | Collect per-token activation dataset | OK | ================= 1049 passed, 12 warnings in 66.16s (0:01:06) ================= |
| 2026-04-06 02:04 UTC | Train Gibbs EBM on per-token activations | OK | ================= 1049 passed, 12 warnings in 66.28s (0:01:06) ================= |
| 2026-04-06 02:11 UTC | Generate 1000+ QA pairs programmatically | OK | ================= 1049 passed, 12 warnings in 66.87s (0:01:06) ================= |
| 2026-04-06 02:19 UTC | MCP server for code verification | OK | ================= 1049 passed, 12 warnings in 66.39s (0:01:06) ================= |
| 2026-04-06 02:25 UTC | CLI tool for code verification | OK | ================= 1049 passed, 12 warnings in 65.56s (0:01:05) ================= |
| 2026-04-09 02:18 UTC | Exp 48: Code → constraint extraction | SKIP | Pre-tests failing: =========== 1 failed, 1107 passed, 12 warnings in 112.26s (0: |
| 2026-04-09 02:39 UTC | Exp 48: Code → constraint extraction | OK | ================ 1130 passed, 12 warnings in 113.04s (0:01:53) ================= |
| 2026-04-09 02:50 UTC | Exp 49: Natural language → constraint extraction | OK | ================ 1130 passed, 12 warnings in 115.46s (0:01:55) ================= |
| 2026-04-09 03:00 UTC | Exp 51: Learn constraint structure from LLM output | OK | ================ 1130 passed, 12 warnings in 112.92s (0:01:52) ================= |
| 2026-04-09 03:11 UTC | Exp 52: Transfer learned Ising across domains | OK | ================ 1130 passed, 12 warnings in 121.06s (0:02:01) ================= |
| 2026-04-09 03:23 UTC | Exp 44: Scheduling constraints | OK | ================ 1130 passed, 12 warnings in 121.05s (0:02:01) ================= |
| 2026-04-09 06:54 UTC | Exp 42c: Fix QUBO carry chain propagation | SKIP | Pre-tests failing: =========== 1 failed, 1129 passed, 12 warnings in 117.99s (0: |
| 2026-04-09 13:54 UTC | Exp 53: Runtime constraint instrumentation | OK | ================ 1130 passed, 12 warnings in 129.64s (0:02:09) ================= |
| 2026-04-09 14:05 UTC | Exp 56: Live LLM → constraint → Ising verification | OK | ================ 1130 passed, 12 warnings in 119.21s (0:01:59) ================= |
| 2026-04-09 14:16 UTC | Exp 57: Live LLM verify-and-repair loop | OK | ================ 1130 passed, 12 warnings in 122.40s (0:02:02) ================= |
| 2026-04-09 14:34 UTC | Exp 59: Constraint-aware prompting | OK | ================ 1130 passed, 12 warnings in 112.45s (0:01:52) ================= |
| 2026-04-09 14:47 UTC | Exp 60: Scale CD training to 100+ variables | OK | ================ 1130 passed, 12 warnings in 121.40s (0:02:01) ================= |
| 2026-04-09 15:21 UTC | Exp 61: Sparse Ising at 500+ variables | OK | ================ 1130 passed, 12 warnings in 130.85s (0:02:10) ================= |
| 2026-04-09 15:31 UTC | Exp 64: Continuous relaxation of Ising constraints | OK | ================ 1130 passed, 12 warnings in 127.37s (0:02:07) ================= |
| 2026-04-09 15:46 UTC | Exp 54: Ising-guided fuzzing | OK | ================ 1130 passed, 12 warnings in 126.64s (0:02:06) ================= |
| 2026-04-09 15:57 UTC | Exp 55: Learn constraints from execution traces | OK | ================ 1130 passed, 12 warnings in 127.27s (0:02:07) ================= |
| 2026-04-09 16:24 UTC | Exp 58: Multi-domain live benchmark (5 domains) | OK | ================ 1130 passed, 12 warnings in 136.29s (0:02:16) ================= |
| 2026-04-09 17:27 UTC | Exp 69: Multi-model verification (Qwen3.5+Gemma4) | OK | ================ 1130 passed, 12 warnings in 175.66s (0:02:55) ================= |
| 2026-04-09 17:36 UTC | Exp 71: Extropic TSU sampler abstraction layer | OK | ================ 1147 passed, 12 warnings in 134.80s (0:02:14) ================= |
| 2026-04-09 17:48 UTC | Exp 73: Constraint coverage metric | OK | ================ 1147 passed, 12 warnings in 126.18s (0:02:06) ================= |
| 2026-04-09 18:08 UTC | Exp 62: Domain-specific constraint learning (10K) | OK | ================ 1147 passed, 12 warnings in 135.06s (0:02:15) ================= |
| 2026-04-09 18:26 UTC | Exp 63: Hierarchical Ising (1000+ vars) | OK | ================ 1147 passed, 12 warnings in 140.56s (0:02:20) ================= |
| 2026-04-09 18:38 UTC | Exp 67: GSM8K subset verification | OK | ================ 1147 passed, 12 warnings in 126.01s (0:02:06) ================= |
| 2026-04-09 18:54 UTC | Exp 68: HumanEval subset verification + fuzzing | OK | ================ 1147 passed, 12 warnings in 131.38s (0:02:11) ================= |
| 2026-04-09 19:10 UTC | Exp 65: Embedding-space constraint verification | OK | ================ 1147 passed, 12 warnings in 128.46s (0:02:08) ================= |
| 2026-04-09 19:22 UTC | Exp 70: Rust constraint extraction + verification | OK | ================ 1147 passed, 12 warnings in 128.80s (0:02:08) ================= |
| 2026-04-09 20:31 UTC | Exp 74: Unified ConstraintExtractor API | OK | ================ 1221 passed, 12 warnings in 201.95s (0:03:21) ================= |
| 2026-04-09 20:49 UTC | Exp 75: VerifyRepairPipeline class | OK | ================ 1263 passed, 12 warnings in 125.45s (0:02:05) ================= |
| 2026-04-09 20:50 UTC | Exp 77: CLI overhaul with pipeline subcommand | OK | Deliverable already exists in repo |
| 2026-04-09 21:37 UTC | Exp 82: Pipeline error handling and edge cases | OK | ================ 1351 passed, 12 warnings in 119.91s (0:01:59) ================= |
| 2026-04-09 21:38 UTC | Exp 76: Production MCP server | OK | Deliverable already exists in repo |
| 2026-04-09 21:59 UTC | Exp 78: PyPI-ready package | OK | ================ 1353 passed, 12 warnings in 127.66s (0:02:07) ================= |
| 2026-04-09 22:09 UTC | Exp 79: Integration examples | OK | ================ 1353 passed, 12 warnings in 115.67s (0:01:55) ================= |
| 2026-04-09 22:18 UTC | Exp 80: Getting started documentation | OK | ================ 1353 passed, 12 warnings in 120.86s (0:02:00) ================= |
| 2026-04-09 22:29 UTC | Exp 81: Integration test suite | OK | ================ 1353 passed, 12 warnings in 118.65s (0:01:58) ================= |
| 2026-04-09 22:36 UTC | Exp 83: Pipeline performance benchmarks | OK | ================ 1353 passed, 12 warnings in 117.63s (0:01:57) ================= |
| 2026-04-09 22:46 UTC | Exp 84: Carnot verifies Carnot | OK | ================ 1353 passed, 12 warnings in 119.71s (0:01:59) ================= |
| 2026-04-09 22:57 UTC | Exp 85: Prepare beta release | OK | ================ 1353 passed, 12 warnings in 121.53s (0:02:01) ================= |
| 2026-04-09 22:58 UTC | Milestone 2026.04.4 activated | OK | 12 tasks queued |
| 2026-04-09 23:18 UTC | Exp 66: End-to-end differentiable constraint reaso | OK | ================= 1353 passed, 12 warnings in 87.24s (0:01:27) ================= |
| 2026-04-09 23:31 UTC | Exp 86: Learned energy composition weights | OK | ================= 1353 passed, 12 warnings in 89.90s (0:01:29) ================= |
| 2026-04-09 23:44 UTC | Exp 87: Gradient-based repair in continuous constr | OK | ================ 1353 passed, 12 warnings in 120.34s (0:02:00) ================= |
| 2026-04-10 00:02 UTC | Exp 88: Failure-driven constraint mining | OK | Deliverable already exists in repo |
| 2026-04-10 00:21 UTC | Exp 89: Self-bootstrapped constraint training | OK | ================ 1376 passed, 12 warnings in 121.93s (0:02:01) ================= |
| 2026-04-10 00:59 UTC | Exp 91: GSM8K live benchmark (Qwen3.5 + Gemma4) | OK | ================ 1376 passed, 12 warnings in 162.22s (0:02:42) ================= |
| 2026-04-10 01:56 UTC | Exp 92: MATH benchmark subset with CoT constraint  | FAIL | Claude error: Another stale background check. All done — Experiment 92 ran |
| 2026-04-10 01:57 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 02:10 UTC | Exp 93: Multi-model systematic comparison | OK | ================ 1376 passed, 12 warnings in 117.64s (0:01:57) ================= |
| 2026-04-10 02:11 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 02:21 UTC | Exp 90: Autoresearch constraint improvement loop | OK | ================ 1376 passed, 12 warnings in 120.64s (0:02:00) ================= |
| 2026-04-10 02:22 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 02:33 UTC | Exp 94: Rust VerifyRepairPipeline | OK | ================ 1376 passed, 12 warnings in 115.40s (0:01:55) ================= |
| 2026-04-10 02:34 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 02:53 UTC | Exp 95: PyO3 pipeline bridge | REVERT | Post-tests failed: ===== 10 failed, 1666 passed, 1 skipped, 12 warnings in 114.4 |
| 2026-04-10 02:54 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 03:03 UTC | Exp 95: PyO3 pipeline bridge | FAIL | Claude error: Error: Reached max turns (50) |
| 2026-04-10 03:04 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 03:04 UTC | Exp 95: PyO3 pipeline bridge | OK | Deliverable already exists in repo |
| 2026-04-10 03:04 UTC | Milestone 2026.04.5 activated | OK | 11 tasks queued |
| 2026-04-10 03:05 UTC | Exp 92: MATH benchmark subset with CoT constraint  | OK | Deliverable already exists in repo |
| 2026-04-10 03:21 UTC | Exp 98: Knowledge-base factual claim verifier | SKIP | Pre-tests failing, self-heal failed: ====== 7 failed, 1477 passed, 1 skipped, 12 |
| 2026-04-10 03:22 UTC | Exp 96: Intermediate result constraint extractor | OK | Deliverable already exists in repo |
| 2026-04-10 03:22 UTC | Exp 97: Comparison constraint extractor | OK | Deliverable already exists in repo |
| 2026-04-10 03:29 UTC | Exp 98: Knowledge-base factual claim verifier | SKIP | Pre-tests failing, self-heal failed: ====== 7 failed, 1477 passed, 1 skipped, 12 |
| 2026-04-10 03:36 UTC | Exp 98: Knowledge-base factual claim verifier | SKIP | Pre-tests failing, self-heal failed: ====== 7 failed, 1477 passed, 1 skipped, 12 |
| 2026-04-10 03:46 UTC | Exp 99: Constraint state propagation across agent  | SKIP | Pre-tests failing, self-heal failed: ====== 6 failed, 1478 passed, 1 skipped, 12 |
| 2026-04-10 04:07 UTC | Exp 99: Constraint state propagation across agent  | SKIP | Pre-tests failing, self-heal failed: = 1384 passed, 1 skipped, 5 xfailed, 95 xpa |
| 2026-04-10 04:17 UTC | Exp 99: Constraint state propagation across agent  | SKIP | Pre-tests failing, self-heal failed: =========== 1484 passed, 1 skipped, 12 warn |
| 2026-04-10 05:04 UTC | Exp 102: Constraint check latency microbenchmark | OK | =========== 1519 passed, 1 skipped, 12 warnings in 120.35s (0:02:00) =========== |
| 2026-04-10 05:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 05:05 UTC | Exp 100: Multi-step verification with rollback | OK | Deliverable already exists in repo |
| 2026-04-10 05:49 UTC | Exp 101: Agent workflow verification end-to-end | OK | =========== 1519 passed, 1 skipped, 12 warnings in 131.15s (0:02:11) =========== |
| 2026-04-10 05:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:20 UTC | Exp 104: Energy-guided token sampling prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:23 UTC | Exp 103: KAN energy tier prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:26 UTC | Exp 103: KAN energy tier prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:29 UTC | Exp 103: KAN energy tier prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:32 UTC | Exp 104: Energy-guided token sampling prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:35 UTC | Exp 104: Energy-guided token sampling prototype | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:36 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:38 UTC | Exp 105: Full-scale benchmark with improved extrac | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:41 UTC | Exp 105: Full-scale benchmark with improved extrac | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:44 UTC | Exp 105: Full-scale benchmark with improved extrac | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:47 UTC | Exp 106: TruthfulQA benchmark with factual extract | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:50 UTC | Exp 106: TruthfulQA benchmark with factual extract | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:53 UTC | Exp 106: TruthfulQA benchmark with factual extract | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:54 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:56 UTC | Exp 107: HuggingFace model card + Exp 66 joint mod | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 06:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 06:59 UTC | Exp 107: HuggingFace model card + Exp 66 joint mod | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:02 UTC | Exp 107: HuggingFace model card + Exp 66 joint mod | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:03 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:03 UTC | Milestone 2026.04.6 activated | OK | 12 tasks queued |
| 2026-04-10 07:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:07 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:08 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:08 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:09 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:10 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:11 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:18 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:21 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:22 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:23 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:24 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:25 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:30 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:31 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:31 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:32 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:32 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:35 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:36 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:40 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:41 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:43 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:44 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:44 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:45 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:46 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:46 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:47 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:47 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:48 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:49 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:49 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:50 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:51 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:52 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:52 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:53 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:53 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:54 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:54 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:55 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:55 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:56 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:56 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:57 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:58 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:58 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 07:59 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 07:59 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:00 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:01 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:01 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:02 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:02 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:03 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:03 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:08 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:09 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:10 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:11 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:16 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:18 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:21 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:22 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:23 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:24 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:25 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:30 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:31 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:32 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:35 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:36 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:40 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:41 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:42 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:43 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:43 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:44 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:44 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:45 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:46 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:46 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:47 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:47 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:48 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:49 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:49 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:50 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:51 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:52 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:52 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:53 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:53 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:54 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:54 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:55 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:55 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:56 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:56 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:58 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 08:59 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 08:59 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:00 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:01 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:01 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:02 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:02 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:03 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:03 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:07 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:08 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:08 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:09 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:10 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:11 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:16 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:18 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:23 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:24 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:25 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:30 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:31 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:31 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:32 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:32 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:35 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:36 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:40 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:41 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:42 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:43 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:43 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:44 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:44 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:46 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:47 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:47 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:48 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:49 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:49 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:50 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:51 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:52 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:52 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:53 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:53 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:54 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:54 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:55 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:55 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:56 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:56 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:57 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:58 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:58 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 09:59 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 09:59 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:00 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:01 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:01 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:02 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:02 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:03 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:03 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:07 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:08 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:08 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:09 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:16 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:18 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:21 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:22 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:23 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:24 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:25 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:30 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:31 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:31 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:32 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:32 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:40 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:41 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:42 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:43 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:43 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:44 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:44 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:45 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:46 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:46 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:47 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:47 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:48 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:49 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:49 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:50 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:51 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:52 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:52 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:53 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:53 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:54 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:54 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:55 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:55 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:56 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:56 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:57 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:58 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:58 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 10:59 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 10:59 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:01 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:02 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:02 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:03 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:03 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:07 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:08 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:08 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:09 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:10 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:11 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:16 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:18 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:21 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:22 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:23 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:24 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:30 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:31 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:31 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:32 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:32 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:33 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:40 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:41 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:43 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:44 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:44 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:45 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:45 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:46 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:46 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:47 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:47 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:48 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:48 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:49 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:49 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:50 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:50 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:51 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:51 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:52 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:52 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:53 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:54 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:55 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:55 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:56 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:56 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:57 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:57 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:58 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:58 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 11:59 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 11:59 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:00 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:01 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:01 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:02 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:03 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:04 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:04 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:05 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:05 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:06 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:07 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:07 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:08 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:09 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:10 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:10 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:11 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:11 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:12 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:12 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:13 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:13 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:14 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:15 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:15 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:16 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:16 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:17 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:19 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:20 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:21 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:21 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:22 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:22 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:23 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:24 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:25 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:25 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:26 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:26 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:27 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:27 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:28 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:28 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:29 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:29 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:30 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:31 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:32 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:33 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:34 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:34 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:35 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:36 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:37 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:37 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:38 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:38 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:39 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:39 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:40 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 12:41 UTC | Plan next milestone | FAIL | Claude error: You've hit your limit · resets 12pm (America/New_York)
 |
| 2026-04-10 12:42 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:14 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:17 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:17 UTC | Plan next milestone | FAIL | Claude error:    at async GeminiClient.processTurn (file:///usr/lib/node_m |
| 2026-04-10 13:18 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:18 UTC | Plan next milestone | FAIL | Claude error:    at async GeminiClient.processTurn (file:///usr/lib/node_m |
| 2026-04-10 13:19 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:19 UTC | Plan next milestone | FAIL | Claude error:    at async GeminiClient.processTurn (file:///usr/lib/node_m |
| 2026-04-10 13:20 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 13:20 UTC | Plan next milestone | FAIL | Claude error:    at async GeminiClient.processTurn (file:///usr/lib/node_m |
| 2026-04-10 14:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 14:00 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 14:05 UTC | Plan milestone 2026.04.7 | OK | 11 tasks proposed |
| 2026-04-10 14:06 UTC | Exp 99: Constraint state propagation across agent  | OK | Deliverable already exists in repo |
| 2026-04-10 14:06 UTC | Milestone 2026.04.7 activated | OK | 11 tasks queued |
| 2026-04-10 14:17 UTC | Exp 108: KAN Energy Function Implementation | OK | =========== 1545 passed, 1 skipped, 12 warnings in 122.25s (0:02:02) =========== |
| 2026-04-10 17:08 UTC | Exp 109: KAN vs Ising vs Gibbs Comparison | OK | Deliverable already exists in repo |
| 2026-04-10 17:49 UTC | Exp 112: Embedding Bottleneck Resolution | REVERT | Post-tests failed: =========== 1567 passed, 1 skipped, 12 warnings in 133.95s (0 |
| 2026-04-10 17:50 UTC | Exp 110: Guided Decoding Prototype Completion | OK | Deliverable already exists in repo |
| 2026-04-10 17:50 UTC | Exp 111: Productionize Guided Decoding | OK | Deliverable already exists in repo |
| 2026-04-10 19:16 UTC | Exp 115: TruthfulQA Benchmark with v12 Extractors | FAIL | Claude Code error:  |
| 2026-04-10 19:17 UTC | Exp 112: Embedding Bottleneck Resolution | OK | Deliverable already exists in repo |
| 2026-04-10 19:17 UTC | Exp 113: Factual Knowledge Base Extractor v2 | OK | Deliverable already exists in repo |
| 2026-04-10 19:17 UTC | Exp 114: WebSearch Fallback for Uncovered Claims | OK | Deliverable already exists in repo |
| 2026-04-10 19:17 UTC | Exp 115: TruthfulQA Benchmark with v12 Extractors | OK | Deliverable already exists in repo |
| 2026-04-10 20:11 UTC | Exp 116: LNN-Based Adaptive Constraint Model | REVERT | Post-tests failed:  |
| 2026-04-10 21:08 UTC | Exp 119: Apple adversarial GSM8K variant generator | OK | =========== 1752 passed, 1 skipped, 12 warnings in 133.77s (0:02:13) =========== |
| 2026-04-10 21:55 UTC | Exp 123: Robust model loading for experiments | OK | =========== 1787 passed, 1 skipped, 12 warnings in 145.38s (0:02:25) =========== |
| 2026-04-10 22:17 UTC | Exp 120: LLM baseline on adversarial GSM8K | OK | =========== 1787 passed, 1 skipped, 12 warnings in 131.94s (0:02:11) =========== |
| 2026-04-10 22:46 UTC | Exp 121: Carnot verify-repair on adversarial GSM8K | FAIL | Claude Code error:  |
| 2026-04-10 22:47 UTC | Exp 121: Carnot verify-repair on adversarial GSM8K | OK | Deliverable already exists in repo |
| 2026-04-10 23:11 UTC | Exp 124: Full GSM8K (1319) with live inference | FAIL | Claude Code error:  |
| 2026-04-10 23:12 UTC | Exp 124: Full GSM8K (1319) with live inference | OK | Deliverable already exists in repo |
| 2026-04-10 23:27 UTC | Exp 122: Adversarial robustness deep analysis | OK | =========== 1787 passed, 1 skipped, 12 warnings in 133.16s (0:02:13) =========== |
| 2026-04-10 23:38 UTC | Exp 125: Constraint state machine for agent workfl | OK | =========== 1813 passed, 1 skipped, 12 warnings in 135.39s (0:02:15) =========== |
| 2026-04-10 23:52 UTC | Exp 126: Agent rollback on constraint violation | OK | =========== 1813 passed, 1 skipped, 12 warnings in 126.72s (0:02:06) =========== |
| 2026-04-11 00:09 UTC | Exp 127: Agent workflow verification benchmark | OK | =========== 1813 passed, 1 skipped, 12 warnings in 125.18s (0:02:05) =========== |
| 2026-04-11 00:36 UTC | Exp 130: Execute adversarial verify-repair | OK | =========== 1844 passed, 1 skipped, 12 warnings in 138.79s (0:02:18) =========== |
| 2026-04-11 01:40 UTC | Exp 134: Online learning benchmark | OK | =========== 1895 passed, 1 skipped, 12 warnings in 134.38s (0:02:14) =========== |
| 2026-04-11 01:41 UTC | Exp 131: Adversarial results analysis and writeup | OK | Deliverable already exists in repo |
| 2026-04-11 01:41 UTC | Exp 132: Constraint performance tracker | OK | Deliverable already exists in repo |
| 2026-04-11 01:41 UTC | Exp 133: Adaptive constraint weighting from tracke | OK | Deliverable already exists in repo |
| 2026-04-11 02:06 UTC | Exp 136: Cross-session learning benchmark | OK | =========== 1928 passed, 1 skipped, 12 warnings in 139.71s (0:02:19) =========== |
| 2026-04-11 02:07 UTC | Exp 135: Persistent constraint memory | OK | Deliverable already exists in repo |
| 2026-04-11 02:17 UTC | Exp 137: Package guided decoding for HuggingFace | OK | =========== 1935 passed, 1 skipped, 12 warnings in 102.88s (0:01:42) =========== |
| 2026-04-11 02:37 UTC | Exp 139: ArXiv research scan and integration | OK | =========== 1935 passed, 1 skipped, 12 warnings in 97.49s (0:01:37) ============ |
| 2026-04-11 02:38 UTC | Exp 138: Guided decoding benchmark | OK | Deliverable already exists in repo |
| 2026-04-11 02:38 UTC | Milestone 2026.04.9 activated | OK | 10 tasks queued |
| 2026-04-11 03:15 UTC | Exp 143: Collect (partial response, final violatio | OK | =========== 1935 passed, 1 skipped, 12 warnings in 96.68s (0:01:36) ============ |
| 2026-04-11 03:16 UTC | Exp 140: Constraint-projection guided decoding lat | OK | Deliverable already exists in repo |
| 2026-04-11 03:42 UTC | Exp 144: Train JEPA violation predictor | OK | =========== 1971 passed, 1 skipped, 12 warnings in 146.39s (0:02:26) =========== |
| 2026-04-11 03:59 UTC | Exp 141: Memory-augmented constraint generation | OK | =========== 2033 passed, 1 skipped, 12 warnings in 149.81s (0:02:29) =========== |
| 2026-04-11 04:13 UTC | Exp 142: Combined Tier 1+2 learning benchmark | OK | =========== 2033 passed, 1 skipped, 12 warnings in 144.79s (0:02:24) =========== |
| 2026-04-11 04:30 UTC | Exp 145: JEPA fast-path / slow-path integration an | OK | =========== 2041 passed, 1 skipped, 12 warnings in 155.41s (0:02:35) =========== |
| 2026-04-11 04:49 UTC | Exp 146: AMD XDNA NPU experimentation | OK | =========== 2041 passed, 1 skipped, 12 warnings in 147.55s (0:02:27) =========== |
| 2026-04-11 05:12 UTC | Exp 147: Apple GSM8K adversarial benchmark — THE c | OK | =========== 2041 passed, 1 skipped, 12 warnings in 186.41s (0:03:06) =========== |
| 2026-04-11 05:41 UTC | Exp 148: Full GSM8K (1319 questions) with live inf | FAIL | Claude Code error:  |
| 2026-04-11 05:42 UTC | Exp 148: Full GSM8K (1319 questions) with live inf | OK | Deliverable already exists in repo |
| 2026-04-11 06:26 UTC | Exp 149: TruthfulQA at scale with factual constrai | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-11 07:04 UTC | Exp 149: TruthfulQA at scale with factual constrai | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-11 08:04 UTC | Exp 149: TruthfulQA at scale with factual constrai | OK | Deliverable already exists in repo |
| 2026-04-11 08:51 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | =========== 2041 passed, 1 skipped, 12 warnings in 342.64s (0:05:42) =========== |
| 2026-04-11 08:52 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | Deliverable already exists in repo |
| 2026-04-11 09:20 UTC | Exp 151: Publish constraint propagation models to  | OK | =========== 2093 passed, 1 skipped, 12 warnings in 379.12s (0:06:19) =========== |
| 2026-04-11 09:21 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | Deliverable already exists in repo |
| 2026-04-11 09:21 UTC | Exp 151: Publish constraint propagation models to  | OK | Deliverable already exists in repo |
| 2026-04-11 09:45 UTC | Exp 152: Continual learning for constraint retenti | OK | =========== 2122 passed, 1 skipped, 12 warnings in 358.11s (0:05:58) =========== |
| 2026-04-11 09:46 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | Deliverable already exists in repo |
| 2026-04-11 09:46 UTC | Exp 151: Publish constraint propagation models to  | OK | Deliverable already exists in repo |
| 2026-04-11 10:10 UTC | Exp 153: KAN adaptive mesh refinement for energy l | OK | =========== 2122 passed, 1 skipped, 12 warnings in 343.12s (0:05:43) =========== |
| 2026-04-11 10:11 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | Deliverable already exists in repo |
| 2026-04-11 10:11 UTC | Exp 151: Publish constraint propagation models to  | OK | Deliverable already exists in repo |
| 2026-04-11 10:11 UTC | Milestone 2026.04.10 activated | OK | 14 tasks queued |
| 2026-04-11 10:12 UTC | Exp 150: Push guided decoding adapter + update 16  | OK | Deliverable already exists in repo |
| 2026-04-11 10:12 UTC | Exp 151: Publish constraint propagation models to  | OK | Deliverable already exists in repo |
| 2026-04-11 11:39 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | =========== 2127 passed, 1 skipped, 12 warnings in 352.62s (0:05:52) =========== |
| 2026-04-11 11:40 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 11:40 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:00 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 13:00 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:00 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 13:01 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 13:01 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:01 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 13:01 UTC | Exp 157: Spilled energy pre-filter for factual hal | OK | Deliverable already exists in repo |
| 2026-04-11 13:02 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 13:02 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:02 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 13:36 UTC | Exp 158: Wikidata-backed factual claim extractor ( | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-11 13:37 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 13:37 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:37 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 13:37 UTC | Exp 158: Wikidata-backed factual claim extractor ( | OK | Deliverable already exists in repo |
| 2026-04-11 13:50 UTC | Exp 159: Full 5-domain benchmark with factual extr | OK | =========== 2229 passed, 1 skipped, 12 warnings in 171.11s (0:02:51) =========== |
| 2026-04-11 13:51 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 13:51 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 13:51 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 14:01 UTC | Exp 160: Connect eGPU (RX 7900 XTX via Thunderbolt | FAIL | Claude Code error: API Error: Unable to connect to API (ConnectionRefused)
 |
| 2026-04-11 14:02 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 14:02 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 14:02 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 14:02 UTC | Exp 160: Connect eGPU (RX 7900 XTX via Thunderbolt | OK | Deliverable already exists in repo |
| 2026-04-11 14:16 UTC | Exp 161: Full GSM8K (1,319 questions) with live in | OK | =========== 2229 passed, 1 skipped, 12 warnings in 162.22s (0:02:42) =========== |
| 2026-04-11 14:17 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 14:17 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 14:17 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 14:46 UTC | Exp 162: Apple adversarial GSM8K with N=200/varian | OK | =========== 2229 passed, 1 skipped, 12 warnings in 174.89s (0:02:54) =========== |
| 2026-04-11 14:47 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 14:47 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 14:47 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 15:25 UTC | Exp 163: HumanEval full 164 problems with live cod | OK | =========== 2251 passed, 1 skipped, 12 warnings in 195.32s (0:03:15) =========== |
| 2026-04-11 15:26 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 15:26 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 15:26 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 15:39 UTC | Exp 164: HuggingFace publishing sprint (guided dec | OK | =========== 2251 passed, 1 skipped, 12 warnings in 197.21s (0:03:17) =========== |
| 2026-04-11 15:40 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 15:40 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 15:40 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 15:50 UTC | Exp 165: ArXiv research scan — prepare next milest | OK | =========== 2251 passed, 1 skipped, 12 warnings in 175.82s (0:02:55) =========== |
| 2026-04-11 15:51 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 15:51 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 15:51 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 15:51 UTC | Milestone 2026.04.11 activated | OK | 12 tasks queued |
| 2026-04-11 15:52 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 15:52 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 15:52 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 16:06 UTC | Plan milestone 2026.04.12 | OK | 14 tasks proposed |
| 2026-04-11 16:07 UTC | Exp 154: Collect multi-domain JEPA training pairs  | OK | Deliverable already exists in repo |
| 2026-04-11 16:07 UTC | Exp 155: Retrain JEPA violation predictor v2 with  | OK | Deliverable already exists in repo |
| 2026-04-11 16:07 UTC | Exp 156: JEPA fast-path benchmark v2 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 16:07 UTC | Milestone 2026.04.12 activated | OK | 14 tasks queued |
| 2026-04-11 16:45 UTC | Exp 166: Logic-aware JEPA training data with symbo | OK | =========== 2251 passed, 1 skipped, 12 warnings in 426.36s (0:07:06) =========== |
| 2026-04-11 17:07 UTC | Exp 167: JEPA predictor v3 — domain-specific symbo | OK | =========== 2256 passed, 1 skipped, 12 warnings in 405.26s (0:06:45) =========== |
| 2026-04-11 17:28 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | =========== 2256 passed, 1 skipped, 12 warnings in 423.79s (0:07:03) =========== |
| 2026-04-11 17:29 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 18:00 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 18:24 UTC | Exp 169: Lookahead energy extractor — AR-EBM bijec | OK | =========== 2294 passed, 1 skipped, 12 warnings in 89.35s (0:01:29) ============ |
| 2026-04-11 18:25 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 18:31 UTC | Exp 170: Real LLM logits benchmark for spilled + l | OK | =========== 2294 passed, 1 skipped, 12 warnings in 92.81s (0:01:32) ============ |
| 2026-04-11 18:32 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 18:43 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | =========== 2294 passed, 1 skipped, 12 warnings in 90.64s (0:01:30) ============ |
| 2026-04-11 18:44 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 18:44 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 19:00 UTC | Exp 172: Global consistency checker for multi-turn | OK | =========== 2328 passed, 1 skipped, 12 warnings in 93.62s (0:01:33) ============ |
| 2026-04-11 19:01 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 19:01 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 19:15 UTC | Exp 173: Constraint generation v2 — NegationConstr | OK | =========== 2386 passed, 1 skipped, 12 warnings in 90.58s (0:01:30) ============ |
| 2026-04-11 19:16 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 19:16 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 19:32 UTC | Exp 174: LagONN constraint solver — escape infeasi | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-11 19:33 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 19:33 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 19:33 UTC | Exp 174: LagONN constraint solver — escape infeasi | OK | Deliverable already exists in repo |
| 2026-04-11 19:52 UTC | Exp 175: Tier-4 KAN adaptive structure self-learni | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-11 19:53 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 19:53 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 19:53 UTC | Exp 175: Tier-4 KAN adaptive structure self-learni | OK | Deliverable already exists in repo |
| 2026-04-11 20:10 UTC | Exp 176: Multi-turn factual verification with glob | OK | =========== 2483 passed, 1 skipped, 22 warnings in 95.28s (0:01:35) ============ |
| 2026-04-11 20:11 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 20:11 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 20:23 UTC | Exp 177: eGPU hardware setup + live model inferenc | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-11 20:24 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 20:24 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 20:24 UTC | Exp 177: eGPU hardware setup + live model inferenc | OK | Deliverable already exists in repo |
| 2026-04-11 20:38 UTC | Exp 178: Definitive adversarial GSM8K — Goal #5 wi | OK | =========== 2483 passed, 1 skipped, 22 warnings in 93.38s (0:01:33) ============ |
| 2026-04-11 20:39 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 20:39 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 20:59 UTC | Exp 179: AMD XDNA NPU activation — VitisAI onnxrun | OK | =========== 2483 passed, 1 skipped, 22 warnings in 108.79s (0:01:48) =========== |
| 2026-04-11 21:00 UTC | Exp 168: JEPA fast-path benchmark v3 — target <2%  | OK | Deliverable already exists in repo |
| 2026-04-11 21:00 UTC | Exp 171: Combined signal pipeline benchmark — all  | OK | Deliverable already exists in repo |
| 2026-04-11 21:00 UTC | Milestone 2026.04.13 activated | OK | 10 tasks queued |
| 2026-04-11 21:23 UTC | Exp 180: GPU inference baseline — Qwen3.5 + Gemma4 | OK | =========== 2484 passed, 1 skipped, 22 warnings in 105.53s (0:01:45) =========== |
| 2026-04-11 21:51 UTC | Exp 181: GSM8K full 1319 with LIVE GPU inference — | FAIL | Claude Code error: e top before any imports
- **No simulation fallback** — `sys |
| 2026-04-11 21:58 UTC | Exp 181: GSM8K full 1319 with LIVE GPU inference — | OK | =========== 2484 passed, 1 skipped, 22 warnings in 115.08s (0:01:55) =========== |
| 2026-04-11 22:22 UTC | Exp 182: Adversarial GSM8K N=400/variant with LIVE | FAIL | Claude Code error:  |
| 2026-04-11 22:36 UTC | Exp 182: Adversarial GSM8K N=400/variant with LIVE | FAIL | Claude Code error:  |
| 2026-04-11 22:51 UTC | Exp 182: Adversarial GSM8K N=400/variant with LIVE | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-11 22:56 UTC | Exp 183: HumanEval full 164 with LIVE GPU — publis | FAIL | Claude Code error:  |
| 2026-04-11 23:21 UTC | Exp 183: HumanEval full 164 with LIVE GPU — publis | FAIL | Claude Code error:  |
| 2026-04-11 23:40 UTC | Exp 183: HumanEval full 164 with LIVE GPU — publis | FAIL | Claude Code error:  |
| 2026-04-12 00:06 UTC | Exp 184: 3B model verification — does verify-repai | FAIL | Claude Code error:  |
| 2026-04-12 02:41 UTC | Exp 203: Extraction autopsy — why ArithmeticExtrac | FAIL | Codex CLI error: 1["correct"] is True
+    assert correct1["failure_category" |
| 2026-04-12 02:42 UTC | Exp 203: Extraction autopsy — why ArithmeticExtrac | OK | Deliverable already exists in repo |
| 2026-04-12 03:08 UTC | Exp 204: Z3 SMT constraint extractor — formal veri | OK | Deliverable already exists in repo |
| 2026-04-12 03:44 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | =========== 1845 passed, 1 skipped, 22 warnings in 164.79s (0:02:44) =========== |
| 2026-04-12 03:45 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 04:47 UTC | Exp 206: Z3 extractor on 100 live GSM8K (Gemma4-E4 | OK | =========== 1854 passed, 1 skipped, 22 warnings in 535.18s (0:08:55) =========== |
| 2026-04-12 04:48 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 05:37 UTC | Exp 207: LLM extractor on 100 live GSM8K — compare | OK | =========== 1858 passed, 1 skipped, 22 warnings in 111.98s (0:01:51) =========== |
| 2026-04-12 05:38 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 06:28 UTC | Exp 208: HumanEval with LIVE IT model — code verif | OK | =========== 1874 passed, 1 skipped, 22 warnings in 112.62s (0:01:52) =========== |
| 2026-04-12 06:29 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 07:00 UTC | Exp 209: Clean up simulation artifacts and update  | OK | =========== 1878 passed, 1 skipped, 22 warnings in 108.75s (0:01:48) =========== |
| 2026-04-12 07:01 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 07:01 UTC | Exp 209: Clean up simulation artifacts and update  | OK | Deliverable already exists in repo |
| 2026-04-12 07:26 UTC | Exp 210: Research scan — focus on constraint extra | OK | =========== 1883 passed, 1 skipped, 22 warnings in 97.65s (0:01:37) ============ |
| 2026-04-12 07:27 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 07:27 UTC | Exp 209: Clean up simulation artifacts and update  | OK | Deliverable already exists in repo |
| 2026-04-12 08:02 UTC | Plan milestone 2026.04.15 | OK | 13 tasks proposed |
| 2026-04-12 08:03 UTC | Exp 205: LLM-as-extractor — use a second LLM call  | OK | Deliverable already exists in repo |
| 2026-04-12 08:03 UTC | Exp 209: Clean up simulation artifacts and update  | OK | Deliverable already exists in repo |
| 2026-04-12 08:03 UTC | Milestone 2026.04.15 activated | OK | 13 tasks queued |
| 2026-04-12 08:33 UTC | Exp 211: Instruction-to-constraint IR benchmark fo | OK | =========== 1889 passed, 1 skipped, 22 warnings in 94.86s (0:01:34) ============ |
| 2026-04-12 09:25 UTC | Exp 213: Chain-of-thought monitorability audit and | OK | =========== 1898 passed, 1 skipped, 22 warnings in 96.61s (0:01:36) ============ |
| 2026-04-12 09:56 UTC | Exp 212: Typed reasoning IR and dual-path extracti | OK | =========== 1907 passed, 1 skipped, 22 warnings in 95.19s (0:01:35) ============ |
| 2026-04-12 10:26 UTC | Exp 214: Semantic failure corpus from live verifie | OK | =========== 1913 passed, 1 skipped, 22 warnings in 98.02s (0:01:38) ============ |
| 2026-04-12 11:15 UTC | Exp 215: Semantic grounding verifier for question- | OK | =========== 1926 passed, 1 skipped, 22 warnings in 98.01s (0:01:38) ============ |
| 2026-04-12 11:40 UTC | Exp 216: Structured reasoning emission path for Qw | OK | =========== 1944 passed, 1 skipped, 22 warnings in 97.32s (0:01:37) ============ |
| 2026-04-12 12:11 UTC | Exp 217: Property-generated code verifier for Huma | OK | =========== 1968 passed, 1 skipped, 22 warnings in 95.49s (0:01:35) ============ |
| 2026-04-12 12:38 UTC | Exp 218: Shared dual-model live benchmark harness | OK | =========== 1977 passed, 1 skipped, 22 warnings in 96.59s (0:01:36) ============ |
| 2026-04-12 14:37 UTC | Exp 219: Live GSM8K semantic benchmark on Qwen3.5- | OK | =========== 1982 passed, 1 skipped, 22 warnings in 86.33s (0:01:26) ============ |
| 2026-04-12 15:19 UTC | Exp 220: Live HumanEval property benchmark on Qwen | OK | =========== 1986 passed, 1 skipped, 22 warnings in 87.76s (0:01:27) ============ |
| 2026-04-12 16:39 UTC | Exp 221: Live prompt-side constraint benchmark on  | OK | =========== 1993 passed, 1 skipped, 22 warnings in 77.86s (0:01:17) ============ |
| 2026-04-12 16:40 UTC | Exp 221: Live prompt-side constraint benchmark on  | OK | Deliverable already exists in repo |
| 2026-04-12 17:13 UTC | Exp 222: Live trace to memory / repair-snippet bui | OK | =========== 2000 passed, 1 skipped, 22 warnings in 77.50s (0:01:17) ============ |
| 2026-04-12 17:14 UTC | Exp 221: Live prompt-side constraint benchmark on  | OK | Deliverable already exists in repo |
| 2026-04-12 17:49 UTC | Exp 223: Chronological replay benchmark for contin | OK | =========== 2006 passed, 1 skipped, 22 warnings in 77.16s (0:01:17) ============ |
| 2026-04-12 17:50 UTC | Exp 221: Live prompt-side constraint benchmark on  | OK | Deliverable already exists in repo |
| 2026-04-12 17:50 UTC | Milestone 2026.04.16 activated | OK | 11 tasks queued |
| 2026-04-12 18:17 UTC | Exp 224: Property-based test generation for code v | OK | =========== 2017 passed, 1 skipped, 22 warnings in 79.65s (0:01:19) ============ |
| 2026-04-12 18:27 UTC | Exp 224a: Warm model server — persistent GPU model | FAIL | Codex CLI error: hmark_cold_load_vs_warm_server
E   ImportError: cannot impor |
| 2026-04-12 19:15 UTC | Exp 224a: Warm model server — persistent GPU model | OK | =========== 2039 passed, 1 skipped, 22 warnings in 83.67s (0:01:23) ============ |
| 2026-04-12 19:44 UTC | Exp 224c: TensorRT-LLM acceleration for 2-4x infer | OK | =========== 2058 passed, 1 skipped, 22 warnings in 80.37s (0:01:20) ============ |
| 2026-04-12 20:40 UTC | Exp 224b: Dual-GPU parallel inference — run both m | OK | =========== 2077 passed, 1 skipped, 22 warnings in 81.68s (0:01:21) ============ |
| 2026-04-12 20:41 UTC | Exp 225: PBT on 30 HumanEval problems — head-to-he | OK | Deliverable already exists in repo |
| 2026-04-12 21:55 UTC | Exp 226: Full 164-problem HumanEval with PBT — pub | OK | =========== 2088 passed, 1 skipped, 22 warnings in 93.06s (0:01:33) ============ |
| 2026-04-12 22:27 UTC | Exp 227: PBT code verification on Qwen3.5-0.8B — c | OK | =========== 2099 passed, 1 skipped, 22 warnings in 84.94s (0:01:24) ============ |
| 2026-04-12 22:59 UTC | Exp 228: KV260 FPGA Ising sampler design and simul | OK | =========== 2120 passed, 1 skipped, 22 warnings in 85.01s (0:01:25) ============ |
| 2026-04-12 23:29 UTC | Exp 229: Self-learning from code verification trac | OK | =========== 2125 passed, 1 skipped, 22 warnings in 86.64s (0:01:26) ============ |
| 2026-04-13 00:10 UTC | Exp 230: Package code verification as standalone t | OK | =========== 2135 passed, 1 skipped, 22 warnings in 92.56s (0:01:32) ============ |
| 2026-04-13 00:32 UTC | Exp 231: Update all documentation with PBT results | OK | =========== 2137 passed, 1 skipped, 22 warnings in 92.42s (0:01:32) ============ |
| 2026-04-13 01:51 UTC | Plan milestone 2026.04.17 | OK | 12 tasks proposed |
| 2026-04-13 01:52 UTC | Milestone 2026.04.17 activated | OK | 12 tasks queued |
| 2026-04-13 02:42 UTC | Exp 232: Semantic verifier calibration corpus from | OK | =========== 2144 passed, 1 skipped, 22 warnings in 88.89s (0:01:28) ============ |
| 2026-04-13 03:36 UTC | Exp 233: Structured-output policy refresh on a JSO | FAIL | Codex CLI error: /python_types.py                   178      0   100%
python/ |
| 2026-04-13 03:36 UTC | Exp 233: Structured-output policy refresh on a JSO | OK | Deliverable already exists in repo |
| 2026-04-13 04:06 UTC | Exp 234: Calibrated semantic verifier v2 | OK | =========== 2163 passed, 1 skipped, 22 warnings in 87.04s (0:01:27) ============ |
| 2026-04-13 05:02 UTC | Exp 235: Live GSM8K semantic benchmark v2 | OK | =========== 2169 passed, 1 skipped, 22 warnings in 87.35s (0:01:27) ============ |
| 2026-04-13 05:31 UTC | Exp 236: Code intent / spec corpus from live Human | FAIL | Codex CLI error:  sys.argv = argv_before
+        monkeypatch.delenv("CARNOT_ |
| 2026-04-13 05:31 UTC | Exp 236: Code intent / spec corpus from live Human | OK | Deliverable already exists in repo |
| 2026-04-13 06:11 UTC | Exp 237: Spec-aware code verifier and repair polic | FAIL | Codex CLI error: ] is True
+    assert status_by_clause[("postconditions", "e |
| 2026-04-13 06:11 UTC | Exp 237: Spec-aware code verifier and repair polic | FAIL | Codex CLI error: e RuntimeError('boom')\n",
+        _PROMPT,
+        "sort_ |
| 2026-04-13 06:11 UTC | Exp 237: Spec-aware code verifier and repair polic | OK | Deliverable already exists in repo |
| 2026-04-13 06:56 UTC | Exp 238: Identical-stack dual-model HumanEval benc | FAIL | Codex CLI error:    assert written["blockers"] == blockers
 
-    module_path |
| 2026-04-13 06:56 UTC | Exp 238: Identical-stack dual-model HumanEval benc | FAIL | Codex CLI error: ilable"
+    assert payload["run_status"] == "partial"
+
+   |
| 2026-04-13 06:56 UTC | Exp 238: Identical-stack dual-model HumanEval benc | FAIL | Codex CLI error: ilable"
+    assert payload["run_status"] == "partial"
+
+   |
| 2026-04-13 06:56 UTC | Exp 238: Identical-stack dual-model HumanEval benc | OK | Deliverable already exists in repo |
| 2026-04-13 07:46 UTC | Exp 239: Case-based trace memory with richer retri | OK | Deliverable already exists in repo |
| 2026-04-13 08:18 UTC | Exp 240: Learned repair-policy compiler from accep | OK | =========== 2216 passed, 1 skipped, 22 warnings in 87.52s (0:01:27) ============ |
| 2026-04-13 08:56 UTC | Exp 241: Chronological self-learning replay v2 | OK | Deliverable already exists in repo |
| 2026-04-13 09:21 UTC | Exp 242: KV260 host / overlay round-trip benchmark | OK | Deliverable already exists in repo |
| 2026-04-13 10:00 UTC | Exp 243: Sampler-guided repair reranking benchmark | OK | Deliverable already exists in repo |
| 2026-04-13 10:35 UTC | Plan milestone 2026.04.18 | OK | 14 tasks proposed |
| 2026-04-13 10:36 UTC | Milestone 2026.04.18 activated | OK | 14 tasks queued |
| 2026-04-13 13:36 UTC | Exp 244: Formal claim-routing corpus from live rea | OK | Deliverable already exists in repo |
| 2026-04-13 14:07 UTC | Exp 245: Solver-routed formal claim verifier | OK | 2300 passed, 1 skipped, 13 warnings in 172.44s (0:02:52) |
| 2026-04-13 14:23 UTC | Exp 246: Live solver-routed semantic benchmark run | OK | 2332 passed, 1 skipped, 13 warnings in 172.48s (0:02:52) |
| 2026-04-13 15:01 UTC | Exp 247: Live solver-routed semantic benchmark | OK | Deliverable already exists in repo |
| 2026-04-13 15:21 UTC | Exp 248: Process-integrity corpus from live semant | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-13 15:22 UTC | Exp 248: Process-integrity corpus from live semant | OK | Deliverable already exists in repo |
| 2026-04-13 15:36 UTC | Exp 249: Additive process verifier for reasoning a | SKIP | Pre-tests failing, self-heal failed: 1 failed, 2352 passed, 2 skipped, 13 warnin |
| 2026-04-13 16:14 UTC | Exp 249: Additive process verifier for reasoning a | OK | 2382 passed, 2 skipped, 13 warnings in 493.48s (0:08:13) |
| 2026-04-13 16:42 UTC | Exp 250: Live process-aware code benchmark runner | OK | 2421 passed, 2 skipped, 13 warnings in 501.75s (0:08:21) |
| 2026-04-13 17:30 UTC | Exp 251: Live process-aware code benchmark | OK | 2421 passed, 2 skipped, 13 warnings in 520.01s (0:08:40) |
| 2026-04-13 17:40 UTC | Exp 252: Predictive verification corpus from parti | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-13 17:41 UTC | Exp 252: Predictive verification corpus from parti | OK | Deliverable already exists in repo |
| 2026-04-13 18:01 UTC | Exp 253: Memory-conditioned constraint addition | OK | 2452 passed, 2 skipped, 13 warnings in 513.57s (0:08:33) |
| 2026-04-13 18:20 UTC | Exp 254: Predictive verifier gate with export-read | OK | 2500 passed, 2 skipped, 13 warnings in 518.65s (0:08:38) |
| 2026-04-13 18:38 UTC | Exp 255: Self-learning A/B benchmark runner | OK | 2533 passed, 2 skipped, 13 warnings in 525.52s (0:08:45) |
| 2026-04-13 19:33 UTC | Exp 256: Self-learning A/B benchmark | OK | Deliverable already exists in repo |
| 2026-04-13 19:50 UTC | Exp 257: Predictive verifier hardware-readiness be | OK | 2533 passed, 1 skipped, 14 warnings in 511.83s (0:08:31) |
| 2026-04-13 20:12 UTC | Plan milestone 2026.04.19 | OK | 13 tasks proposed |
| 2026-04-13 20:13 UTC | Milestone 2026.04.19 activated | OK | 13 tasks queued |
| 2026-04-13 20:47 UTC | Exp 258: Wire DualGPURunner to live benchmark harn | OK | 2567 passed, 2 skipped, 14 warnings in 505.40s (0:08:25) |
| 2026-04-13 21:04 UTC | Exp 259: onnxruntime-gpu CUDA EP unlock and Predic | OK | 2579 passed, 5 skipped, 13 warnings in 487.77s (0:08:07) |
| 2026-04-13 22:18 UTC | Exp 260: Complete solver-routed semantic benchmark | OK | Deliverable already exists in repo |
| 2026-04-13 22:21 UTC | Exp 261: Full 164-problem HumanEval benchmark with | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:25 UTC | Exp 261: Full 164-problem HumanEval benchmark with | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:30 UTC | Exp 261: Full 164-problem HumanEval benchmark with | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-13 22:34 UTC | Exp 262: Live calibration corpus for PredictiveVer | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:38 UTC | Exp 262: Live calibration corpus for PredictiveVer | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:42 UTC | Exp 262: Live calibration corpus for PredictiveVer | SKIP | Pre-tests failing, self-heal failed: 8 failed, 81 passed in 3.41s |
| 2026-04-13 22:46 UTC | Exp 263: Calibrate PredictiveVerifier and run cali | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:47 UTC | Exp 262: Live calibration corpus for PredictiveVer | OK | Deliverable already exists in repo |
| 2026-04-13 22:50 UTC | Exp 263: Calibrate PredictiveVerifier and run cali | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 22:54 UTC | Exp 263: Calibrate PredictiveVerifier and run cali | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-13 22:58 UTC | Exp 264: Domain-specific constraint template extra | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:02 UTC | Exp 264: Domain-specific constraint template extra | SKIP | Pre-tests failing, self-heal failed: 81 passed, 12 errors in 3.35s |
| 2026-04-13 23:07 UTC | Exp 264: Domain-specific constraint template extra | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:08 UTC | Exp 265: Constraint addition module wired to forma | OK | Deliverable already exists in repo |
| 2026-04-13 23:11 UTC | Exp 266: Self-learning replay v3 with calibrated g | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:12 UTC | Exp 266: Self-learning replay v3 with calibrated g | OK | Deliverable already exists in repo |
| 2026-04-13 23:15 UTC | Exp 267: Update 16 HuggingFace model READMEs | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:22 UTC | Exp 267: Update 16 HuggingFace model READMEs | FAIL | Post-tests failed:  |
| 2026-04-13 23:23 UTC | Exp 267: Update 16 HuggingFace model READMEs | OK | Deliverable already exists in repo |
| 2026-04-13 23:26 UTC | Exp 268: Publish Exp 66 joint model and FormalClai | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:30 UTC | Exp 268: Publish Exp 66 joint model and FormalClai | SKIP | Pre-tests failing, self-heal failed: 17 failed, 81 passed, 5 skipped in 3.37s |
| 2026-04-13 23:34 UTC | Exp 268: Publish Exp 66 joint model and FormalClai | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:38 UTC | Exp 269: AMD XDNA NPU enablement for PredictiveVer | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:42 UTC | Exp 269: AMD XDNA NPU enablement for PredictiveVer | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:46 UTC | Exp 269: AMD XDNA NPU enablement for PredictiveVer | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:51 UTC | Exp 270: Operational retrospective for milestone 2 | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:55 UTC | Exp 270: Operational retrospective for milestone 2 | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-13 23:59 UTC | Exp 270: Operational retrospective for milestone 2 | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-14 00:08 UTC | Plan next milestone | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-14 00:09 UTC | Exp 270: Operational retrospective for milestone 2 | OK | Deliverable already exists in repo |
| 2026-04-14 00:16 UTC | Plan next milestone | FAIL | Claude Code error: Stalled after 180s silence. Last output:  |
| 2026-04-14 00:36 UTC | Plan next milestone | FAIL | Claude Code error: Stalled after 600s silence. Last output:  |
| 2026-04-14 00:52 UTC | Plan next milestone | FAIL | Claude Code error: Stalled after 600s silence. Last output:  |
| 2026-04-14 00:56 UTC | Milestone 2026.04.20 activated | OK | 10 tasks queued |
| 2026-04-14 01:07 UTC | Exp 271: Global consistency checker on live multi- | SKIP | Pre-tests failing, self-heal failed: 1 failed, 80 passed in 3.20s |
| 2026-04-14 01:22 UTC | Exp 271: Global consistency checker on live multi- | FAIL | Claude Code error: Stalled after 600s silence. Last output:  |
| 2026-04-14 01:23 UTC | Exp 271: Global consistency checker on live multi- | OK | Deliverable already exists in repo |
| 2026-04-14 01:43 UTC | Exp 272: Self-learning Tier 1 retrained on live tr | FAIL | Post-tests failed:  |
| 2026-04-14 01:53 UTC | Exp 272: Self-learning Tier 1 retrained on live tr | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 01:54 UTC | Exp 272: Self-learning Tier 1 retrained on live tr | OK | Deliverable already exists in repo |
| 2026-04-14 02:05 UTC | Exp 273: Agent rollback verification on live model | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 02:06 UTC | Exp 273: Agent rollback verification on live model | OK | Deliverable already exists in repo |
| 2026-04-14 02:30 UTC | Exp 274: Factual extractor (Wikidata) on live IT m | FAIL | Post-tests failed:  |
| 2026-04-14 02:46 UTC | Exp 274: Factual extractor (Wikidata) on live IT m | FAIL | Post-tests failed:  |
| 2026-04-14 02:47 UTC | Exp 274: Factual extractor (Wikidata) on live IT m | OK | Deliverable already exists in repo |
| 2026-04-14 03:22 UTC | Exp 275: Adaptive KAN verification with live trace | OK | Deliverable already exists in repo |
| 2026-04-14 03:38 UTC | Exp 276: Full GSM8K with Z3+LLM+semantic extractor | OK | 3002 passed, 26 skipped, 13 warnings in 172.75s (0:02:52) |
| 2026-04-14 03:53 UTC | Exp 277: Combined verification signals with modern | OK | 3068 passed, 26 skipped, 13 warnings in 172.87s (0:02:52) |
| 2026-04-14 04:06 UTC | Exp 278: Cross-session constraint memory with live | OK | 3084 passed, 26 skipped, 13 warnings in 176.58s (0:02:56) |
| 2026-04-14 04:22 UTC | Exp 279: Adversarial number-swapped GSM8K with sem | OK | 3100 passed, 26 skipped, 13 warnings in 174.95s (0:02:54) |
| 2026-04-14 04:30 UTC | Exp 280: Revalidation sweep summary and docs updat | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 04:37 UTC | Exp 280: Revalidation sweep summary and docs updat | OK | 3100 passed, 26 skipped, 13 warnings in 175.33s (0:02:55) |
| 2026-04-14 04:59 UTC | Plan milestone 2026.04.21 | OK | 14 tasks proposed |
| 2026-04-14 05:00 UTC | Milestone 2026.04.21 activated | OK | 14 tasks queued |
| 2026-04-14 05:13 UTC | Exp 281: Apple adversarial GSM8K dataset generator | OK | 3112 passed, 26 skipped, 13 warnings in 171.92s (0:02:51) |
| 2026-04-14 05:28 UTC | Exp 282: Apple adversarial GSM8K GPU baseline (sav | OK | 3128 passed, 26 skipped, 13 warnings in 176.25s (0:02:56) |
| 2026-04-14 05:43 UTC | Exp 283: Apple adversarial GSM8K + verify-repair — | OK | 3151 passed, 26 skipped, 13 warnings in 178.98s (0:02:58) |
| 2026-04-14 05:56 UTC | Exp 284: Apple adversarial results analysis and do | OK | 3182 passed, 26 skipped, 13 warnings in 179.22s (0:02:59) |
| 2026-04-14 06:06 UTC | Exp 285: SpilledEnergyExtractor implementation (ar | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 06:07 UTC | Exp 285: SpilledEnergyExtractor implementation (ar | OK | Deliverable already exists in repo |
| 2026-04-14 06:19 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 06:20 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 06:47 UTC | Exp 287: Dual-energy benchmark on Apple adversaria | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 06:48 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 06:48 UTC | Exp 287: Dual-energy benchmark on Apple adversaria | OK | Deliverable already exists in repo |
| 2026-04-14 07:00 UTC | Exp 288: KV260 FPGA overlay bring-up validation | OK | 3302 passed, 28 skipped, 13 warnings in 180.01s (0:03:00) |
| 2026-04-14 07:01 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 07:15 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 07:16 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 07:16 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 07:32 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | 3376 passed, 28 skipped, 13 warnings in 186.84s (0:03:06) |
| 2026-04-14 07:33 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 07:33 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 07:40 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | 3376 passed, 28 skipped, 13 warnings in 187.75s (0:03:07) |
| 2026-04-14 07:41 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 07:41 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 07:41 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 07:50 UTC | Exp 291: JEPA predictor retrained on Apple adversa | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 07:51 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 07:51 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 07:51 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 07:59 UTC | Exp 291: JEPA predictor retrained on Apple adversa | OK | 3423 passed, 28 skipped, 13 warnings in 188.84s (0:03:08) |
| 2026-04-14 08:00 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 08:00 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 08:00 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 08:15 UTC | Exp 292: AMD XDNA NPU enablement — onnxruntime sou | OK | 3442 passed, 39 skipped, 13 warnings in 187.20s (0:03:07) |
| 2026-04-14 08:16 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 08:16 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 08:16 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 08:34 UTC | Exp 293: HuggingFace — Exp 66 joint model and Form | OK | 3484 passed, 39 skipped, 13 warnings in 191.16s (0:03:11) |
| 2026-04-14 08:35 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 08:35 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 08:35 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 08:48 UTC | Exp 294: Operational retrospective for milestone 2 | OK | 3519 passed, 39 skipped, 13 warnings in 186.01s (0:03:06) |
| 2026-04-14 08:49 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 08:49 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 08:49 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 09:07 UTC | Plan milestone 2026.04.22 | OK | 13 tasks proposed |
| 2026-04-14 09:08 UTC | Exp 286: SemanticEnergyExtractor + DualEnergyGate  | OK | Deliverable already exists in repo |
| 2026-04-14 09:08 UTC | Exp 289: FpgaBackend with quantum-inspired sparse  | OK | Deliverable already exists in repo |
| 2026-04-14 09:08 UTC | Exp 290: FPGA vs CPU Ising benchmark (hardware or  | OK | Deliverable already exists in repo |
| 2026-04-14 09:08 UTC | Milestone 2026.04.22 activated | OK | 13 tasks queued |
| 2026-04-14 09:28 UTC | Exp 294: GPU stall diagnosis + Apple adversarial b | FAIL | Post-tests failed: 12 failed, 3523 passed, 39 skipped, 13 warnings in 202.05s (0 |
| 2026-04-14 09:40 UTC | Exp 294: GPU stall diagnosis + Apple adversarial b | OK | 3535 passed, 39 skipped, 13 warnings in 207.71s (0:03:27) |
| 2026-04-14 09:56 UTC | Exp 295: Apple adversarial verify-repair re-run (w | OK | 3564 passed, 39 skipped, 13 warnings in 209.77s (0:03:29) |
| 2026-04-14 10:07 UTC | Exp 296: Apple adversarial analysis, classificatio | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 10:08 UTC | Exp 296: Apple adversarial analysis, classificatio | OK | Deliverable already exists in repo |
| 2026-04-14 10:08 UTC | Exp 297: SemanticEnergyExtractor + VarEntropyProbe | OK | Deliverable already exists in repo |
| 2026-04-14 10:30 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | 3649 passed, 39 skipped, 13 warnings in 212.11s (0:03:32) |
| 2026-04-14 10:31 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 10:42 UTC | Exp 299: JEPA retrain on real Apple adversarial lo | OK | 3700 passed, 39 skipped, 13 warnings in 210.20s (0:03:30) |
| 2026-04-14 10:43 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 10:59 UTC | Exp 300: Memory-to-Constraint Generator — Tier 2 p | OK | 3741 passed, 39 skipped, 13 warnings in 212.10s (0:03:32) |
| 2026-04-14 11:00 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 11:14 UTC | Exp 301: Confidence-weighted constraint violations | OK | 3779 passed, 39 skipped, 13 warnings in 212.70s (0:03:32) |
| 2026-04-14 11:15 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 11:35 UTC | Exp 302: Self-learning integrated benchmark — cons | OK | 3846 passed, 39 skipped, 13 warnings in 213.23s (0:03:33) |
| 2026-04-14 11:36 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 11:51 UTC | Exp 303: AMD XDNA NPU VitisAI unblock — install ni | OK | 3862 passed, 53 skipped, 13 warnings in 212.78s (0:03:32) |
| 2026-04-14 11:52 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 12:05 UTC | Exp 304: HuggingFace actual publish — run Exp 293  | OK | 3886 passed, 54 skipped, 13 warnings in 211.34s (0:03:31) |
| 2026-04-14 12:06 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 12:06 UTC | Exp 304: HuggingFace actual publish — run Exp 293  | OK | Deliverable already exists in repo |
| 2026-04-14 12:21 UTC | Exp 305: KV260 Verilog Ising sampler — first synth | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-14 12:22 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 12:22 UTC | Exp 304: HuggingFace actual publish — run Exp 293  | OK | Deliverable already exists in repo |
| 2026-04-14 12:22 UTC | Exp 305: KV260 Verilog Ising sampler — first synth | OK | Deliverable already exists in repo |
| 2026-04-14 12:42 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 12:42 UTC | Exp 304: HuggingFace actual publish — run Exp 293  | OK | Deliverable already exists in repo |
| 2026-04-14 12:42 UTC | Exp 306: Operational efficiency — scaffolding temp | OK | Deliverable already exists in repo |
| 2026-04-14 13:00 UTC | Plan milestone 2026.04.23 | OK | 13 tasks proposed |
| 2026-04-14 13:01 UTC | Exp 298: PrefillUncertaintyProbe — pre-generation  | OK | Deliverable already exists in repo |
| 2026-04-14 13:01 UTC | Exp 304: HuggingFace actual publish — run Exp 293  | OK | Deliverable already exists in repo |
| 2026-04-14 13:01 UTC | Milestone 2026.04.23 activated | OK | 13 tasks queued |
| 2026-04-14 13:45 UTC | Exp 307: JEPA train on real Apple adversarial logi | OK | Deliverable already exists in repo |
| 2026-04-14 16:02 UTC | Exp 308: JEPA fast-path gate integration and laten | FAIL | Post-tests failed:  |
| 2026-04-14 16:03 UTC | Exp 308: JEPA fast-path gate integration and laten | OK | Deliverable already exists in repo |
| 2026-04-14 17:08 UTC | Exp 309: Tier 3 end-to-end self-learning — real lo | FAIL | Post-tests failed:  |
| 2026-04-14 17:09 UTC | Exp 309: Tier 3 end-to-end self-learning — real lo | OK | Deliverable already exists in repo |
| 2026-04-14 17:48 UTC | Exp 310: NL2Z3Extractor — LLM-translated Z3 SMT as | FAIL | Post-tests failed:  |
| 2026-04-14 17:49 UTC | Exp 310: NL2Z3Extractor — LLM-translated Z3 SMT as | OK | Deliverable already exists in repo |
| 2026-04-14 18:23 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | FAIL | Post-tests failed:  |
| 2026-04-14 18:24 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 19:11 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 19:11 UTC | Exp 312: Z3-gated repair — only repair when Z3 say | OK | Deliverable already exists in repo |
| 2026-04-14 19:53 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 19:53 UTC | Exp 313: KV260 FPGA actual bring-up — PYNQ overlay | OK | Deliverable already exists in repo |
| 2026-04-14 20:14 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 20:14 UTC | Exp 314: NPU prereq install + unblock retry (ninja | OK | Deliverable already exists in repo |
| 2026-04-14 20:22 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | 81 passed in 3.24s |
| 2026-04-14 20:23 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 20:23 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 20:30 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | 81 passed in 3.23s |
| 2026-04-14 20:31 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 20:31 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 20:31 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 20:59 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 20:59 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 20:59 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 21:01 UTC | Exp 317: HuggingFace publish — update all model RE | OK | 81 passed in 3.19s |
| 2026-04-14 21:02 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 21:02 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 21:02 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 21:28 UTC | Exp 318: Continuous self-learning relay — Tiers 1+ | OK | 81 passed in 3.25s |
| 2026-04-14 21:29 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 21:29 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 21:29 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 21:57 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 21:57 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 21:57 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 22:01 UTC | Exp 322: Reward hacking detection in self-learning | OK | 81 passed in 3.03s |
| 2026-04-14 22:02 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 22:02 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 22:02 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 22:29 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 22:29 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 22:29 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 22:48 UTC | Exp 323: Conductor behavioral audit log with anoma | OK | 81 passed in 3.24s |
| 2026-04-14 22:49 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 22:49 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 22:49 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 22:57 UTC | Exp 324: Conductor constitution — explicit rules f | OK | 100 passed in 3.39s |
| 2026-04-14 22:58 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 22:58 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 22:58 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 23:30 UTC | Exp 320: D-Wave sampler backend with local Neal si | OK | 97 passed in 4.99s |
| 2026-04-14 23:31 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 23:31 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 23:31 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-14 23:59 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-14 23:59 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-14 23:59 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-15 00:32 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-15 00:32 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-15 00:32 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-15 00:32 UTC | Exp 321: D-Wave Neal vs CPU Ising on constraint ve | OK | Deliverable already exists in repo |
| 2026-04-15 00:39 UTC | Exp 319: Operational retrospective for milestone 2026.04.23 | OK | operational_retro_2026_04_23.json written; n=17 experiments, total=691 min, top bottleneck=Exp 308 (138.0 min); RETRO-001/002 carried forward; NEW-001/002 added; estimated next-milestone speedup ~15% |
| 2026-04-15 01:04 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-15 01:04 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-15 01:04 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-15 01:04 UTC | Exp 319: Operational retrospective — milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-15 01:26 UTC | Plan milestone 2026.04.24 | OK | 13 tasks proposed |
| 2026-04-15 01:27 UTC | Exp 311: Extractor benchmark — regex vs LLM vs Z3  | OK | Deliverable already exists in repo |
| 2026-04-15 01:27 UTC | Exp 315: Full-scale benchmark script — GSM8K 400q  | OK | Deliverable already exists in repo |
| 2026-04-15 01:27 UTC | Exp 316: Full-scale benchmark execution — run Exp  | OK | Deliverable already exists in repo |
| 2026-04-15 01:27 UTC | Milestone 2026.04.24 activated | OK | 13 tasks queued |
| 2026-04-15 01:51 UTC | Exp 325: Conductor hardening — RETRO-001 + NEW-001 | OK | run_experiment_with_timeout.sh written (45 min default, CARNOT_CONDUCTOR_TIMEOUT_MINUTES); generate_test_stub() added to ExperimentTemplate; 23 tests pass; RETRO-001 implemented (carried forward 2× milestones); NEW-001 implemented; estimated speedup 27% |

## Retrospective Action Items

### RETRO-001: Conductor timeout ≤ 45 min
- **Status:** IMPLEMENTED (Exp 325, 2026-04-15)
- **Root cause:** `claude -p` called with no timeout; stuck experiments run to max 50 turns (~2+ hours)
- **Fix:** `scripts/run_experiment_with_timeout.sh` wraps any command with `timeout -k 60s ${CARNOT_CONDUCTOR_TIMEOUT_MINUTES:-45}m`
- **Evidence:** Exp 308 consumed 138 min; 45-min cap saves ≥93 min per stuck experiment
- **Usage:** `CARNOT_CONDUCTOR_TIMEOUT_MINUTES=30 ./scripts/run_experiment_with_timeout.sh python scripts/research_conductor.py`

### RETRO-002: GPU monitor integration (gpu_monitor_results in setup_gpu)
- **Status:** IMPLEMENTED (Exp 326, 2026-04-15)
- **Root cause:** `ExperimentTemplate.setup_gpu()` pre-warmed models but never checked for zombie processes
  holding VRAM at 0% utilisation (PIDs 2592400/2595103, ~1050 MB each, observed in 2026.04.23 retro).
- **Fix:** `DualGPUMonitor` (`python/carnot/pipeline/dual_gpu_monitor.py`) runs `nvidia-smi` queries at
  experiment start; `setup_gpu()` now returns an additive `gpu_monitor_results` key with
  `n_gpus_detected`, `n_zombies`, `idle_gpus`, `all_healthy`.  Warnings logged when
  `CARNOT_FORCE_LIVE=1`.  CI-safe: FileNotFoundError returns `[]` without raising.
- **Evidence:** Zombie processes silently consumed ~1050 MB and were invisible to the scaffolding.
- **Spec:** REQ-INFRA-003, SCENARIO-INFRA-004, SCENARIO-INFRA-006

### RETRO-003: DualGPURunner enforcement (idle GPU detection)
- **Status:** IMPLEMENTED (Exp 326, 2026-04-15)
- **Root cause:** Exp 219/221 ran two models sequentially on GPU 0 while GPU 1 sat idle for the entire
  run (~195 min; estimated ~90 min if both GPUs used in parallel — ~105 min wasted).
- **Fix:** `DualGPUMonitor.check_dual_gpu_health()` identifies idle GPU indices (GPUs with zero active
  processes) and includes them in the artifact.  `all_healthy=True` only when `n_gpus_detected≥2`,
  `n_zombies==0`, and `idle_gpus==[]`.
- **Evidence:** `results/operational_retro_2026_04_23.json` `items_carried_forward` field confirms
  this was carried forward from the 2026.04.22 milestone.
- **Spec:** REQ-INFRA-004, SCENARIO-INFRA-005, SCENARIO-INFRA-006

### NEW-001: Test-first stub generation
- **Status:** IMPLEMENTED (Exp 325, 2026-04-15)
- **Root cause:** 23.5% post-test failure rate in 2026.04.23 milestone partly from tests written after implementation
- **Fix:** `ExperimentTemplate.generate_test_stub(test_file_path, module_to_test)` writes a pytest skeleton before implementation; idempotent
- **Usage:** Call `tmpl.generate_test_stub("tests/python/test_exp_NNN.py", "scripts.my_module")` at start of experiment setup

### NEW-002: Pre-experiment dependency audit
- **Status:** IMPLEMENTED (Exp 327, 2026-04-15)
- **Root cause:** Experiments fail mid-run when they assume prior experiment results exist
  but those files are missing (e.g., "load results/experiment_307_jepa_real_training.json").
  ~5% wall-time overhead observed from retry loops across the 2026.04.23 milestone.
- **Fix:** `scripts/experiment_dependency_audit.py` — parses "EXISTING CODE TO READ FIRST:"
  section from any research prompt, resolves each file path, and reports missing files before
  the experiment begins.  Exit code 0 = all present; code 1 = missing files (with list).
  `check_dependencies(prompt, project_root)` returns `DependencyAudit` dataclass; conductor
  can call `build_blocked_artifact(audit)` to emit a blocked artifact without spending any
  inference tokens.  `load_experiment_prompt(yaml_path, exp_id)` loads a prompt from a
  roadmap YAML by matching exp_id substring in the task id field.
- **Evidence:** `results/experiment_327_dep_audit_results.json` — validated against first 3
  prompts in research-roadmap.yaml; correctly detected research-roadmap-next.yaml as missing
  from Exp 327's own dependency list.
- **Spec:** REQ-INFRA-005, SCENARIO-INFRA-007, SCENARIO-INFRA-008
| 2026-04-15 02:02 UTC | Exp 325: Conductor timeout wrapper + ExperimentTem | OK | 81 passed in 3.28s |
| 2026-04-15 02:04 UTC | Exp 326: DualGPUMonitor — RETRO-002 + RETRO-003    | OK | 32 tests pass; DualGPUMonitor + GPUProcessInfo in pipeline; setup_gpu() additive gpu_monitor_results key; RETRO-002/003 implemented |
| 2026-04-15 02:28 UTC | Exp 326: DualGPUMonitor + ExperimentTemplate GPU e | OK | 81 passed in 3.53s |
| 2026-04-15 02:29 UTC | Exp 327: Pre-experiment dependency audit (NEW-002)  | OK | 34 tests pass; DependencyAudit + extract_required_files + check_dependencies + build_blocked_artifact + load_experiment_prompt + CLI; artifact: results/experiment_327_dep_audit_results.json; REQ-INFRA-005 implemented |
| 2026-04-15 02:57 UTC | Exp 327: Pre-experiment dependency audit tool (NEW | OK | Deliverable already exists in repo |
| 2026-04-15 03:24 UTC | Exp 328: Live GPU full-scale benchmark — run Exp 3 | OK | Deliverable already exists in repo |
| 2026-04-15 03:52 UTC | Exp 329: Four-tier self-learning relay on live GPU | OK | Deliverable already exists in repo |
| 2026-04-15 04:17 UTC | Exp 330: HuggingFace live publish — run Exp 317 sc | OK | 81 passed in 3.76s |
| 2026-04-15 04:26 UTC | Exp 331: False positive autopsy — categorize broke | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 04:27 UTC | Exp 331: False positive autopsy — categorize broke | OK | Deliverable already exists in repo |
| 2026-04-15 04:59 UTC | Exp 332: Confidence-weighted constraint violations | OK | 177 passed in 3.67s |
| 2026-04-15 05:30 UTC | Exp 333: Model-adaptive constraint thresholds + se | OK | 81 passed in 3.18s |
| 2026-04-15 05:39 UTC | Exp 334: VERGE-style iterative Z3 refinement — tar | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 05:59 UTC | Exp 334: VERGE-style iterative Z3 refinement — tar | OK | 175 passed in 3.77s |
| 2026-04-15 06:09 UTC | Exp 335: AMD XDNA NPU build — install prereqs and  | OK | 81 passed in 3.22s |
| 2026-04-15 06:10 UTC | Exp 335: AMD XDNA NPU build — install prereqs and  | OK | Deliverable already exists in repo |
| 2026-04-15 06:20 UTC | Exp 336: CoTCircuitVerifier — CRV-style chain-of-t | OK | 145 passed in 3.74s |
| 2026-04-15 06:21 UTC | Exp 335: AMD XDNA NPU build — install prereqs and  | OK | Deliverable already exists in repo |
| 2026-04-15 06:29 UTC | Exp 337: Operational retrospective for milestone 2026.04.24 | OK | operational_retro_2026_04_24.json written; n=12 experiments, total=293 min, mean=24.4 min/exp, top bottleneck=Exp 325 (35.0 min); all 4 RETRO items resolved; NEW-003/004 added; actual speedup ~39.9%; estimated next speedup ~4.0% |
| 2026-04-15 06:52 UTC | Exp 335: AMD XDNA NPU build — install prereqs and  | OK | Deliverable already exists in repo |
| 2026-04-15 06:52 UTC | Exp 337: Operational retrospective for milestone 2 | OK | Deliverable already exists in repo |
| 2026-04-15 07:14 UTC | Plan milestone 2026.04.25 | OK | 13 tasks proposed |
| 2026-04-15 07:15 UTC | Exp 335: AMD XDNA NPU build — install prereqs and  | OK | Deliverable already exists in repo |
| 2026-04-15 07:15 UTC | Milestone 2026.04.25 activated | OK | 13 tasks queued |
| 2026-04-15 07:46 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 08:10 UTC | Exp 339: Pre-session startup health check (RETRO-007/008) | OK | 63 tests pass; session_startup.sh + session_startup.py; parse_session_startup_output + run_session_startup; RETRO-007 + RETRO-008 implemented; REQ-INFRA-008 |
| 2026-04-15 08:18 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 08:18 UTC | Exp 339: Pre-session GPU health + zombie cleanup a | OK | Deliverable already exists in repo |
| 2026-04-15 08:52 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | 81 passed in 3.25s |
| 2026-04-15 08:53 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 09:14 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | 81 passed in 3.36s |
| 2026-04-15 09:15 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 09:16 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 09:18 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 09:18 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 09:48 UTC | Exp 341: Live HumanEval code verification — CodeEx | OK | 81 passed in 3.21s |
| 2026-04-15 09:49 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 09:49 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 10:31 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 10:31 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 10:31 UTC | Exp 342: Live extractor comparison — ArithmeticExt | OK | Deliverable already exists in repo |
| 2026-04-15 11:08 UTC | Exp 343: ConstraintTemplateLibrary — error pattern | OK | 145 passed in 25.97s |
| 2026-04-15 11:09 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 11:09 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 11:55 UTC | Exp 344: Constraint addition benchmark — CaseMemor | OK | 171 passed in 20.08s |
| 2026-04-15 11:56 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 11:56 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 12:30 UTC | Exp 345: Multi-session memory persistence — CaseMe | OK | 145 passed in 27.53s |
| 2026-04-15 12:31 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 12:31 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 13:30 UTC | Exp 346: EORM-style energy reward model — train 55 | OK | 81 passed in 23.38s |
| 2026-04-15 13:31 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 13:31 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 13:56 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 13:56 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:21 UTC | Exp 347: JEPA predictor retraining on real violati | OK | 81 passed in 12.96s |
| 2026-04-15 14:22 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:22 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:35 UTC | Exp 348: SinkProbe attention-sink hallucination pr | OK | 81 passed in 27.34s |
| 2026-04-15 14:36 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:36 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:41 UTC | Exp 349: KV260 FPGA bitfile synthesis via open-sou | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:42 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:42 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:43 UTC | Exp 349: KV260 FPGA bitfile synthesis via open-sou | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:44 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:44 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:45 UTC | Exp 349: KV260 FPGA bitfile synthesis via open-sou | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:46 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:46 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:46 UTC | Exp 350: Operational retrospective for milestone 2 | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:47 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:47 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:48 UTC | Exp 350: Operational retrospective for milestone 2 | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:49 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:49 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:51 UTC | Plan next milestone | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:52 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:52 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:55 UTC | Plan next milestone | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:56 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:56 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:58 UTC | Plan next milestone | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 14:59 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 14:59 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 14:59 UTC | Exp 350: Operational retrospective for milestone 2 | FAIL | Claude Code error: API Error: 500 {"type":"error","error":{"type":"api_error"," |
| 2026-04-15 15:00 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 15:00 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 15:20 UTC | Plan milestone 2026.04.26 | OK | 13 tasks proposed |
| 2026-04-15 15:21 UTC | Exp 338: Host prereqs registry + DualGPURunner as  | OK | Deliverable already exists in repo |
| 2026-04-15 15:21 UTC | Exp 340: Live full precision pipeline benchmark —  | OK | Deliverable already exists in repo |
| 2026-04-15 15:21 UTC | Exp 350: Operational retrospective for milestone 2 | OK | Deliverable already exists in repo |
| 2026-04-15 15:21 UTC | Milestone 2026.04.26 activated | OK | 13 tasks queued |
| 2026-04-15 15:22 UTC | Exp 351: Close RETRO-003/005/009/010/011 — conduct | OK | Deliverable already exists in repo |
| 2026-04-15 15:33 UTC | Exp 352: Live GPU diagnostic — identify failure layer | OK | LiveGPUDiagnostic implemented; diagnose_live_gpu() 100% coverage; setup_gpu() now raises RuntimeError on CARNOT_FORCE_LIVE=1 failure |
| 2026-04-15 15:33 UTC | Exp 352: Live GPU inference root-cause diagnostic  | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 15:37 UTC | Exp 352: Live GPU diagnostic — diagnostic findings review | OK | Confirmed: live_gpu_diagnostic.py 100% coverage (37 tests pass); setup_gpu() raises RuntimeError when CARNOT_FORCE_LIVE=1+prewarm fails; root-cause fix for silent simulated fallback in Exps 340/341/346/347; experiment_352 script + spec REQ-INFRA-014/SCENARIO-INFRA-014/015 all in place |
| 2026-04-15 15:39 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | 118 passed in 19.86s |
| 2026-04-15 15:43 UTC | Exp 352: Live GPU inference root-cause diagnostic  | FAIL | No file changes produced |
| 2026-04-15 15:50 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | 81 passed in 10.62s |
| 2026-04-15 15:54 UTC | Exp 352: Live GPU diagnostic RUN — root cause confirmed | OK | RESULT: is_live_capable=True. All layers pass: cuda_visible=True, torch_cuda=True, model_loadable=True (Qwen3.5-0.8B + gemma-4-E4B-it). ROOT CAUSE: carnot_force_live_set=False — CARNOT_FORCE_LIVE=1 was NOT propagated into conductor subprocess environment when Exps 340/341/346/347 ran. GPU hardware is fine. Fix: conductor must pass CARNOT_FORCE_LIVE=1 in subprocess env when launching live GPU experiments. artifact: results/experiment_352_live_gpu_diagnostic.json |
| 2026-04-15 15:56 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | 81 passed in 8.72s |
| 2026-04-15 15:57 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 16:34 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 16:34 UTC | Exp 353: Live GPU smoke test — 5 questions, verify | OK | Deliverable already exists in repo |
| 2026-04-15 16:48 UTC | Exp 354: Apple adversarial GSM8K harness — dataset | OK | 81 passed in 8.73s |
| 2026-04-15 16:49 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 17:16 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | 81 passed in 9.24s |
| 2026-04-15 17:17 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |

| 2026-04-15 17:19 UTC | Exp 355: Apple adversarial GSM8K benchmark — live GPU execution on Gemma4-E4B-it + Qwen3.5-0.5B | OK | honest_verdict=blocked_simulated (CARNOT_FORCE_LIVE not set); 51 tests pass; artifact: results/experiment_355_adversarial_gsm8k_benchmark.json; live GPU execution requires CARNOT_FORCE_LIVE=1 with pre-warmed models |
| 2026-04-15 17:20 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | 81 passed in 8.68s |
| 2026-04-15 17:21 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 17:21 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 17:21 UTC | Exp 356: LLMExtractor — second LLM call extracts s | OK | Deliverable already exists in repo |
| 2026-04-15 17:46 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 17:46 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 17:46 UTC | Exp 357: LLM-guided Z3 formalization — arXiv 2601. | OK | Deliverable already exists in repo |
| 2026-04-15 17:57 UTC | Exp 358: Comparative extraction benchmark — ArithmeticExtractor vs LLMConstraintExtractor vs LLMz3Formalizer on 50 Gemma4-E4B-it responses | OK | scripts/experiment_358_extraction_benchmark.py + python/carnot/pipeline/extraction_benchmark.py + 33 tests all pass |
| 2026-04-15 18:11 UTC | Exp 358: Extraction benchmark — ArithmeticExtracto | OK | 81 passed in 7.26s |
| 2026-04-15 18:12 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 18:12 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 18:21 UTC | Exp 359: EORM retrain on real (CoT, correctness) p | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 18:22 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 18:22 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 18:45 UTC | Exp 359: EORM retrain on real (CoT, correctness) p | OK | 129 passed in 3.32s |
| 2026-04-15 18:46 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 18:46 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 19:22 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 19:22 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 19:23 UTC | Exp 360: Three-Tier Pipeline Benchmark — SinkProbe+EORM+Ising vs Ising-alone | OK | 54 tests pass; total_skip_rate=0.80, fn_rate=0.71 (EORM AUC=0.5); results/experiment_360_three_tier_benchmark.json written; cpu_synthetic mode |
| 2026-04-15 19:25 UTC | Exp 360: Three-tier pipeline benchmark — SinkProbe | OK | 81 passed in 3.07s |
| 2026-04-15 19:26 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 19:26 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 19:55 UTC | Exp 361: Tier 1+2+3 online self-learning relay — r | OK | 81 passed in 3.34s |
| 2026-04-15 19:56 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 19:56 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 20:08 UTC | Exp 362: SAVeR multi-turn verification — constrain | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 20:09 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 20:09 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 20:09 UTC | Exp 362: SAVeR multi-turn verification — constrain | OK | Deliverable already exists in repo |
| 2026-04-15 20:55 UTC | Exp 364: Wire ModelServer + TensorRT + DualGPU int | OK | 112 passed in 3.30s |
| 2026-04-15 20:56 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 20:56 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 21:40 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 21:40 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 21:40 UTC | Exp 363: Operational retrospective for milestone 2 | OK | Deliverable already exists in repo |
| 2026-04-15 21:56 UTC | Plan milestone 2026.04.27 | OK | 12 tasks proposed |
| 2026-04-15 21:57 UTC | Exp 352: Live GPU inference root-cause diagnostic  | OK | Deliverable already exists in repo |
| 2026-04-15 21:57 UTC | Exp 355: Apple adversarial GSM8K benchmark — live  | OK | Deliverable already exists in repo |
| 2026-04-15 21:57 UTC | Milestone 2026.04.27 activated | OK | 12 tasks queued |
| 2026-04-15 22:06 UTC | Exp 365: Close RETRO-012/013/014 — conductor GPU e | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 22:08 UTC | Exp 365: Close RETRO-012/013/014 — conductor GPU env fix + JSON enforcer | OK | 73 tests pass; all_closed=True; scripts/conductor_gpu_env.sh written; missing_jsons=[357,358,362] flagged for follow-up; results/experiment_365_retro_close.json written |
| 2026-04-15 22:08 UTC | RETRO-012 | CLOSED | scripts/conductor_gpu_env.sh created with export CARNOT_FORCE_LIVE=1; source before GPU experiments |
| 2026-04-15 22:08 UTC | RETRO-013 | CLOSED | Exp 356 gap documented; addressed by Exp 366 this milestone |
| 2026-04-15 22:08 UTC | RETRO-014 | CLOSED | RetroJSONEnforcer pattern enforced; missing JSONs 357/358/362 flagged for human follow-up |
| 2026-04-15 22:10 UTC | Exp 365: Close RETRO-012/013/014 — conductor GPU e | OK | 154 passed in 3.36s |
| 2026-04-15 22:11 UTC | Exp 366: LLMExtractor — use a small LLM call to ex | OK | Deliverable already exists in repo |
| 2026-04-15 22:39 UTC | Exp 367: Live extraction comparison — ArithmeticExtractor vs LLMConstraintExtractor vs LLMz3Formalizer on Gemma4-E4B-it GSM8K output | OK | 42 tests pass; ExtractorComparisonResult + run_extractor_comparison + build_extractor_comparison_artifact added to python/carnot/pipeline/extractor_comparison.py; experiment script + tests written; honest_verdict=live_gpu_winner only when CARNOT_FORCE_LIVE=1 + all extractors live_gpu; blocked artifact when CARNOT_FORCE_LIVE not set |
| 2026-04-15 22:43 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | 81 passed in 3.30s |
| 2026-04-15 23:03 UTC | Exp 367: Human-requested verification — Exp 367 + Exp 358 tests | OK | 75 passed in 11.93s; full suite 6577 passed (80 pre-existing failures in test_experiment_319_retro.py unrelated); REQ-EXTRACT-023 + SCENARIO-EXTRACT-047/048 confirmed in spec; all ops docs updated |
| 2026-04-15 23:05 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | 81 passed in 3.31s |
| 2026-04-15 23:29 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | 81 passed in 3.26s |
| 2026-04-15 23:36 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-15 23:50 UTC | Exp 368: Live precision pipeline benchmark — 200 G | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-15 23:51 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-15 23:54 UTC | Exp 368: Live precision pipeline benchmark — 200 G | OK | 155 passed in 3.31s |
| 2026-04-15 23:55 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 00:22 UTC | Exp 369: Live HumanEval code verification — 50 pro | OK | 81 passed in 3.36s |
| 2026-04-16 00:23 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 00:50 UTC | Exp 370: Live adversarial GSM8K benchmark — Carnot credibility experiment | OK | 23 tests pass; diagnose_live_gpu_or_raise hard gate (no simulated fallback); LLMConstraintExtractor for repair; adversarial_schema=carnot.adversarial_gsm8k.v2; SCENARIO-BENCH-022 added; 6742 pass full suite; LIVE RUN PENDING CARNOT_FORCE_LIVE=1 |
| 2026-04-16 00:52 UTC | Exp 370: Live adversarial GSM8K — Apple arXiv 2410 | OK | 81 passed in 3.37s |
| 2026-04-16 00:53 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 01:17 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 01:46 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 01:46 UTC | Exp 371: EORM real-data retrain — train on live pa | OK | Deliverable already exists in repo |
| 2026-04-16 02:01 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 02:01 UTC | Exp 372: JEPA real-data retrain — train violation  | OK | Deliverable already exists in repo |
| 2026-04-16 02:21 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 02:21 UTC | Exp 372: JEPA real-data retrain — train violation  | OK | Deliverable already exists in repo |
| 2026-04-16 02:21 UTC | Exp 373: Three-tier pipeline on live GPU — SinkPro | OK | Deliverable already exists in repo |
| 2026-04-16 02:37 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 02:37 UTC | Exp 372: JEPA real-data retrain — train violation  | OK | Deliverable already exists in repo |
| 2026-04-16 02:37 UTC | Exp 374: Self-learning relay on live GPU — FR-11 f | OK | Deliverable already exists in repo |
| 2026-04-16 02:37 UTC | Exp 375: CIKAN constraint-informed KAN energy tier | OK | Deliverable already exists in repo |
| 2026-04-16 02:56 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 02:56 UTC | Exp 372: JEPA real-data retrain — train violation  | OK | Deliverable already exists in repo |
| 2026-04-16 02:56 UTC | Exp 376: Operational retrospective — milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-16 03:19 UTC | Plan milestone 2026.04.28 | OK | 13 tasks proposed |
| 2026-04-16 03:20 UTC | Exp 367: Live extraction benchmark — LLMExtractor  | OK | Deliverable already exists in repo |
| 2026-04-16 03:20 UTC | Exp 372: JEPA real-data retrain — train violation  | OK | Deliverable already exists in repo |
| 2026-04-16 03:20 UTC | Milestone 2026.04.28 activated | OK | 13 tasks queued |
| 2026-04-16 04:09 UTC | Exp 377: Fix RETRO-015 — session_startup.sh auto-s | OK | Deliverable already exists in repo |
| 2026-04-16 04:09 UTC | Exp 378: Fix RETRO-018 — CIKANEnergy proper Python | OK | Deliverable already exists in repo |
| 2026-04-16 04:41 UTC | Exp 379: Live precision pipeline execution — 200 G | OK | Deliverable already exists in repo |
| 2026-04-16 05:42 UTC | Exp 380: Live HumanEval code verification executio | OK | Deliverable already exists in repo |
| 2026-04-16 05:42 UTC | Exp 381: Live adversarial GSM8K execution — Carnot | OK | Deliverable already exists in repo |
| 2026-04-16 05:42 UTC | Exp 382: Live extraction comparison — LLMExtractor | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 383: Combined EORM + JEPA retrain on live pair | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 384: FR-11 self-learning relay live — first le | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 385: Three-tier pipeline live — SinkProbe + EO | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 386: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 387: Safety/Jailbreak KAN Classifier — first T | OK | Deliverable already exists in repo |
| 2026-04-16 06:45 UTC | Exp 388: SAVeR live multi-turn verification — fait | OK | Deliverable already exists in repo |
| 2026-04-16 07:22 UTC | Exp 386: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 07:22 UTC | Exp 389: Operational retrospective — milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-16 07:43 UTC | Plan milestone 2026.04.29 | OK | 14 tasks proposed |
| 2026-04-16 07:44 UTC | Exp 386: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 07:44 UTC | Milestone 2026.04.29 activated | OK | 14 tasks queued |
| 2026-04-16 08:21 UTC | Exp 390: GPU node preflight — confirm live GPU bef | OK | Deliverable already exists in repo |
| 2026-04-16 08:21 UTC | Exp 391: Fix RETRO-020 — CIKANEnergy proper Python | OK | Deliverable already exists in repo |
| 2026-04-16 08:21 UTC | Exp 392: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 08:21 UTC | Exp 393: Safety/Jailbreak KAN Classifier — first T | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 392: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 394: Live precision pipeline — 200 GSM8K, 5 va | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 395: Live HumanEval code verification — 50 pro | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 396: Live adversarial GSM8K — Carnot's headlin | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 397: Live extraction comparison — LLMExtractor | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 398: Combined EORM+JEPA retrain on live pairs  | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 399: FR-11 self-learning relay live — first le | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 400: SAVeR live multi-turn verification — fait | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 401: Semantic Energy hallucination scorer — lo | OK | Deliverable already exists in repo |
| 2026-04-16 09:22 UTC | Exp 402: CRANE extraction gate — alternating free  | OK | Deliverable already exists in repo |
| 2026-04-16 09:40 UTC | Exp 392: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 09:40 UTC | Exp 398: Combined EORM+JEPA retrain on live pairs  | OK | Deliverable already exists in repo |
| 2026-04-16 09:40 UTC | Exp 402: CRANE extraction gate — alternating free  | OK | Deliverable already exists in repo |
| 2026-04-16 09:40 UTC | Exp 403: Operational retrospective — milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-16 10:02 UTC | Plan milestone 2026.04.30 | OK | 14 tasks proposed |
| 2026-04-16 10:03 UTC | Exp 392: JitRL constraint memory — non-parametric  | OK | Deliverable already exists in repo |
| 2026-04-16 10:03 UTC | Exp 398: Combined EORM+JEPA retrain on live pairs  | OK | Deliverable already exists in repo |
| 2026-04-16 10:03 UTC | Exp 402: CRANE extraction gate — alternating free  | OK | Deliverable already exists in repo |
| 2026-04-16 10:03 UTC | Milestone 2026.04.30 activated | OK | 14 tasks queued |
| 2026-04-16 10:46 UTC | Exp 404: DeliverableContentValidator + GPU preflig | OK | Deliverable already exists in repo |
| 2026-04-16 10:46 UTC | Exp 405: CIKANEnergy — third and final attempt wit | OK | Deliverable already exists in repo |
| 2026-04-16 10:46 UTC | Exp 406: JitRL constraint memory — correct Tier 1  | OK | Deliverable already exists in repo |
| 2026-04-16 10:46 UTC | Exp 407: Safety KAN classifier — first Tier B prod | OK | Deliverable already exists in repo |
| 2026-04-16 10:46 UTC | Exp 408: Semantic Energy scorer — logit-space Bolt | OK | Deliverable already exists in repo |
| 2026-04-16 10:46 UTC | Exp 409: CRANE extraction gate — 1x-cost prompt-si | OK | Deliverable already exists in repo |
| 2026-04-16 11:51 UTC | Exp 410: Live precision pipeline — 200 GSM8K, 5 va | OK | 81 passed in 3.43s |
| 2026-04-16 11:52 UTC | Exp 406: JitRL constraint memory — correct Tier 1  | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 406: JitRL constraint memory — correct Tier 1  | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 411: Live HumanEval code verification — 50 pro | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 412: Live adversarial GSM8K — Carnot's headlin | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 413: Live extraction comparison — CRANE vs LLM | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 414: EORM+JEPA retrain on live pairs from Exps | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 415: FR-11 self-learning relay live — close RE | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 416: MathAgent constraint graph builder — LLME | OK | Deliverable already exists in repo |
| 2026-04-16 12:36 UTC | Exp 417: Operational retrospective — milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-16 12:56 UTC | Plan milestone 2026.04.31 | OK | 12 tasks proposed |
| 2026-04-16 12:57 UTC | Exp 406: JitRL constraint memory — correct Tier 1  | OK | Deliverable already exists in repo |
| 2026-04-16 12:57 UTC | Milestone 2026.04.31 activated | OK | 12 tasks queued |
| 2026-04-16 13:45 UTC | Exp 413: EnvironmentAutoFix — self-configuring CAR | OK | 81 passed in 3.39s |
| 2026-04-16 13:46 UTC | Exp 414: CIKANEnergy — constraint-informed KAN wit | OK | Deliverable already exists in repo |
| 2026-04-16 13:46 UTC | Exp 415: JitRL constraint memory — threshold modul | OK | Deliverable already exists in repo |
| 2026-04-16 13:46 UTC | Exp 416: Safety KAN classifier — first Tier B prod | OK | Deliverable already exists in repo |
| 2026-04-16 13:46 UTC | Exp 417: Semantic Energy Scorer — logit-space Bolt | OK | Deliverable already exists in repo |
| 2026-04-16 13:46 UTC | Exp 418: CRANE extraction gate — 1x-cost prompt-si | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 419: Live precision pipeline — 200 GSM8K, 5 va | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 420: Live HumanEval code verification — 50 pro | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 421: Live adversarial GSM8K — Carnot's headlin | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 422: VPRM training via FOVER — Z3-annotated GS | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 423: EORM + JEPA retrain on live data + FR-11  | OK | Deliverable already exists in repo |
| 2026-04-16 15:36 UTC | Exp 424: Operational retrospective milestone 2026. | OK | Deliverable already exists in repo |
| 2026-04-16 16:16 UTC | Exp 423: EORM + JEPA retrain on live data + FR-11  | OK | Deliverable already exists in repo |
| 2026-04-16 16:39 UTC | Plan milestone 2026.04.32 | OK | 12 tasks proposed |
| 2026-04-16 16:50 UTC | Exp 423: EORM + JEPA retrain on live data + FR-11  | OK | Deliverable already exists in repo |
| 2026-04-16 16:51 UTC | Milestone 2026.04.32 activated | OK | 12 tasks queued |
| 2026-04-16 18:33 UTC | Exp 425: Conductor timeout watchdog — RETRO-003 (1 | OK | 81 passed in 3.38s |
| 2026-04-16 18:46 UTC | Exp 426: DualGPU Fix + Temp Guard — RETRO-025    | OK | 35 passed in 9.26s |
| 2026-04-16 18:46 UTC | RETRO-025: GPU1 zombie detection implemented       | OK | check_dual_gpu_health + temp guard in setup_gpu() |
| 2026-04-16 20:35 UTC | Exp 426: DualGPURunner GPU-1 scheduling fix + temp | FAIL | Claude Code error: Full suite result: **7734 passed** (up from 7733 in the prev |
| 2026-04-16 20:52 UTC | Exp 426: DualGPU Fix + Temp Guard — RETRO-025 verified | OK | 35 passed in 10.82s — RETRO-025 CLOSED |
| 2026-04-16 21:37 UTC | Exp 426: DualGPURunner GPU-1 scheduling fix + temp | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-16 21:42 UTC | Exp 426: DualGPURunner GPU-1 scheduling fix + temp | OK | Deliverable already exists in repo |
| 2026-04-16 22:27 UTC | Exp 427: Live precision benchmark — confirm or re- | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-16 22:32 UTC | Exp 427: Live precision benchmark — confirm or re- | OK | Deliverable already exists in repo |
| 2026-04-16 23:17 UTC | Exp 428: Live HumanEval code verification — confir | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-16 23:22 UTC | Exp 428: Live HumanEval code verification — confir | OK | Deliverable already exists in repo |
| 2026-04-17 00:08 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 00:14 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | FAIL | Claude Code error:  |
| 2026-04-17 00:19 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 01:04 UTC | Exp 430: FOVER-style Z3 step annotation — automati | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 01:09 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 01:09 UTC | Exp 430: FOVER-style Z3 step annotation — automati | OK | Deliverable already exists in repo |
| 2026-04-17 01:55 UTC | Exp 431: EORM + JEPA retrain on FOVER real-data pa | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 02:00 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 02:47 UTC | Exp 431: EORM + JEPA retrain on FOVER real-data pa | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-17 02:52 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 02:52 UTC | Exp 431: EORM + JEPA retrain on FOVER real-data pa | OK | Deliverable already exists in repo |
| 2026-04-17 02:52 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 02:54 UTC | Exp 432: JitRL Constraint Memory — Live Validation | OK | 39 passed — synthetic_fallback (Exp 427 scaffolding_only) |
| 2026-04-17 03:39 UTC | Exp 432: JitRL live validation — Tier 1 self-learn | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 03:44 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 04:30 UTC | Exp 432: JitRL live validation — Tier 1 self-learn | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 04:35 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 04:35 UTC | Exp 432: JitRL live validation — Tier 1 self-learn | OK | Deliverable already exists in repo |
| 2026-04-17 04:44 UTC | Exp 433: SpilledEnergyDetector benchmark — arXiv 2602.18671 per-token logit-discrepancy | OK | SpilledEnergyDetector + SpilledEnergyDetectorResult + SpilledEnergyToken + compute_detector_spilled_energy; CI text mode; Tier 0 in ThreeTierPipeline |
| 2026-04-17 05:15 UTC | Exp 433: Spilled Energy pre-filter — logit-discrep | OK | 81 passed in 38.92s |
| 2026-04-17 05:20 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 05:37 UTC | Exp 434: Compliance Checker — Tier B product for r | OK | 81 passed in 19.95s |
| 2026-04-17 05:42 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 05:43 UTC | Exp 435: AMD XDNA NPU Unblock — 5th attempt + IRON toolchain probe | OK | 50 tests pass; blocked_prereq (5th milestone); IRON path documented |
| 2026-04-17 05:55 UTC | Exp 435: AMD XDNA NPU unblock — prereq install che | OK | 81 passed in 16.38s |
| 2026-04-17 06:00 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 06:07 UTC | Exp 435a: Kona-adjacent continuous energy landscap | OK | 81 passed in 19.82s |
| 2026-04-17 06:12 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 06:58 UTC | Exp 436: Operational Retrospective — Milestone 202 | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 07:03 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 07:03 UTC | Exp 436: Operational Retrospective — Milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-17 07:26 UTC | Plan milestone 2026.04.33 | OK | 13 tasks proposed |
| 2026-04-17 07:31 UTC | Exp 429: Live adversarial GSM8K benchmark — Apple  | OK | Deliverable already exists in repo |
| 2026-04-17 07:31 UTC | Milestone 2026.04.33 activated | OK | 13 tasks queued |
| 2026-04-17 07:48 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 CLOSED | OK | LongRunBenchmarkExecutor + BenchmarkBatch + LongRunBenchmarkResult + get_batch_size; 25 tests pass 100% module coverage; results/experiment_437_long_run_executor.json; REQ-INFRA-027/028, SCENARIO-INFRA-034/035/036 |
| 2026-04-17 08:22 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 08:27 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 09:12 UTC | Exp 438: GPU1 zombie root-cause fix — DualGPURunne | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 09:17 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 09:19 UTC | Exp 438: GPU1 zombie root-cause fix — DualGPURunne | FAIL | No file changes produced |
| 2026-04-17 09:24 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 09:24 UTC | Exp 438: GPU1 zombie root-cause fix — DualGPURunne | OK | Deliverable already exists in repo |
| 2026-04-17 10:09 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 10:14 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 11:00 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 11:05 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 11:05 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 11:50 UTC | Exp 440: Live HumanEval Micro-Benchmark — 50 probl | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 11:55 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 11:55 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 12:40 UTC | Exp 440: Live HumanEval Micro-Benchmark — 50 probl | FAIL | Claude Code error: Wall-clock timeout after 2703s. Last output:  |
| 2026-04-17 12:45 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 12:45 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 12:45 UTC | Exp 440: Live HumanEval Micro-Benchmark — 50 probl | OK | Deliverable already exists in repo |
| 2026-04-17 13:30 UTC | Exp 441: Live Adversarial GSM8K Micro — 50q x 3 co | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 13:35 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 13:35 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 14:20 UTC | Exp 441: Live Adversarial GSM8K Micro — 50q x 3 co | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 14:25 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 14:25 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 14:25 UTC | Exp 441: Live Adversarial GSM8K Micro — 50q x 3 co | OK | Deliverable already exists in repo |
| 2026-04-17 14:28 UTC | Exp 442: FOVER live CoT annotation (FR-11 upstream)  | OK | fover_live.py + experiment_442 script + 28 tests pass; Exp 439 live CoT (300 responses, inference_mode=live_gpu confirmed) ready for annotation; honest_verdict will be real_data_labeled when executed; FR-11 relay unblocked |
| 2026-04-17 15:11 UTC | Exp 442: FOVER live annotation — Z3 step labels on | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 15:16 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 15:16 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 16:01 UTC | Exp 442: FOVER live annotation — Z3 step labels on | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 16:06 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 16:06 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 16:06 UTC | Exp 442: FOVER live annotation — Z3 step labels on | OK | Deliverable already exists in repo |
| 2026-04-17 16:43 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 16:43 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 16:51 UTC | Exp 443: EORM+JEPA live retrain on Exp 442 real pa | FAIL | Claude Code error: Wall-clock timeout after 2702s. Last output:  |
| 2026-04-17 16:56 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 16:56 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 17:43 UTC | Exp 443: EORM+JEPA live retrain on Exp 442 real pa | SKIP | Pre-tests failing, self-heal failed:  |
| 2026-04-17 17:49 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 17:49 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 17:49 UTC | Exp 443: EORM+JEPA live retrain on Exp 442 real pa | OK | Deliverable already exists in repo |
| 2026-04-17 18:03 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 18:03 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 19:03 UTC | Exp 444: CarnotThinkProbe — ThinkPRM-style CoT ver | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-17 19:08 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 19:08 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 20:09 UTC | Exp 444: CarnotThinkProbe — ThinkPRM-style CoT ver | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-17 20:14 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 20:14 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 20:14 UTC | Exp 444: CarnotThinkProbe — ThinkPRM-style CoT ver | OK | Deliverable already exists in repo |
| 2026-04-17 21:14 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-17 21:19 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 21:19 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 22:02 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 22:02 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 22:02 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-17 23:03 UTC | Exp 446: Energy Matching for ContinuousEBM — Phase | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-17 23:08 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 23:08 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 23:08 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-17 23:11 UTC | Exp 446: Energy Matching for ContinuousEBM — Phase | FAIL | Claude Code error: [Errno 122] Disk quota exceeded |
| 2026-04-17 23:16 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-17 23:16 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-17 23:16 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 00:39 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-18 00:39 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-18 00:39 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 01:40 UTC | Exp 446: Energy Matching for ContinuousEBM — Phase | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 02:10 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-18 02:10 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-18 02:10 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 03:10 UTC | Exp 447: KAEMEnergy — KAN exact inverse-transform  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 03:40 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-18 03:40 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-18 03:40 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 04:41 UTC | Exp 447: KAEMEnergy — KAN exact inverse-transform  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 05:11 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-18 05:11 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-18 05:11 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 06:11 UTC | Exp 447: KAEMEnergy — KAN exact inverse-transform  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 06:41 UTC | Exp 437: LongRunBenchmarkExecutor — RETRO-026 fix  | OK | Deliverable already exists in repo |
| 2026-04-18 06:41 UTC | Exp 439: Live Precision Micro-Benchmark — 50q x 3  | OK | Deliverable already exists in repo |
| 2026-04-18 06:41 UTC | Exp 445: BoltzmannRepairBridge — DBM energy → LLM  | OK | Deliverable already exists in repo |
| 2026-04-18 07:42 UTC | Exp 447: KAEMEnergy — KAN exact inverse-transform  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 08:10 UTC | Exp 447: KAEMEnergy — KAN exact inverse-transform | OK | Deliverable already exists in repo |
| 2026-04-18 09:10 UTC | Exp 448: Tier 2 Cross-Session Constraint Memory Re | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 09:40 UTC | Exp 448: Tier 2 Cross-Session Constraint Memory Re | OK | Deliverable already exists in repo |
| 2026-04-18 09:41 UTC | Exp 449: Milestone 2026.04.33 Retrospective — FIRST live GPU numbers confirmed (honest negatives); RETRO-024/026 CLOSED; RETRO-028/029/030/031 OPENED; 75 tests pass | OK | operational_retro_2026_04_33.json written |
| 2026-04-18 10:41 UTC | Exp 449: Operational Retrospective — Milestone 202 | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 11:11 UTC | Exp 449: Operational Retrospective — Milestone 202 | OK | Deliverable already exists in repo |
| 2026-04-18 11:35 UTC | Plan milestone 2026.04.34 | OK | 12 tasks proposed |
| 2026-04-18 11:36 UTC | Milestone 2026.04.34 activated | OK | 12 tasks queued |
| 2026-04-18 11:42 UTC | Exp 450: RETRO-028 Fix — GemmaTransformersLoader (llama.cpp#21516 workaround) | OK | apply_env_autofix() FIRST; ExperimentTimeoutWatchdog(450, 30min); new module python/carnot/pipeline/gemma_loader.py: GemmaTransformersLoader(model_id, device='auto') with load() via AutoModelForCausalLM.from_pretrained (NOT llama.cpp), generate(prompt, max_new_tokens=512) → str, is_valid_output(text) → bool (rejects all-<unusedN>-token strings via regex); ValueError if model_id doesn't contain 'gemma'/'Gemma'; detailed WHY docstrings: llama.cpp#21516 bug description, token_id=14 explanation, silent failure rationale; REQ-LOADER-001/002 + SCENARIO-LOADER-001/002 added to verifiable-reasoning/spec.md; exported GemmaTransformersLoader from carnot.pipeline.__init__; scripts/experiment_450_gemma4_fix.py: apply_env_autofix() FIRST, ExperimentTimeoutWatchdog(450, 30min), ExperimentTemplate(450), 10 hardcoded GSM8K questions, gpu_required/error/success paths, honest_verdict retro_028_fix_ready/retro_028_verified, schema=carnot.gemma_loader.v1; 20 tests pass (test_gemma_loader.py); RETRO-028 fix implemented — GemmaTransformersLoader ready for GPU verification run |
| 2026-04-18 12:42 UTC | Exp 450: Gemma4 Tokenizer Fix — RETRO-028 closure  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 13:47 UTC | Exp 450: Gemma4 Tokenizer Fix — RETRO-028 closure  | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-18 13:57 UTC | Exp 450: Gemma4 Tokenizer Fix — RETRO-028 closure  | OK | 81 passed in 18.74s |
| 2026-04-18 14:13 UTC | Exp 451: Live Precision Re-Run Post-Fix — first po | OK | 81 passed in 16.92s |
| 2026-04-18 15:19 UTC | Exp 452: Energy Matching v2 — RETRO-030 closure (E | FAIL | Claude Code error: Wall-clock timeout after 3604s. Last output:  |
| 2026-04-18 16:26 UTC | Exp 452: Energy Matching v2 — RETRO-030 closure (E | FAIL | Claude Code error: Wall-clock timeout after 3604s. Last output:  |
| 2026-04-18 16:31 UTC | Exp 452: Energy Matching v2 — RETRO-030 closure (E | OK | Deliverable already exists in repo |
| 2026-04-18 16:51 UTC | Exp 453: VeriCoT Step Validator — FOL formalizatio | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-18 16:58 UTC | Exp 453: VeriCoT Step Validator — FOL formalization + Z3 UNSAT detection | OK | 56 tests pass; ArithmeticExtractor=0, VeriCoT=8/20 (improvement_rate=0.40), honest_verdict=vericot_better |
| 2026-04-18 17:46 UTC | Exp 453: VeriCoT Step Validator — FOL formalizatio | OK | Deliverable already exists in repo |
| 2026-04-18 18:10 UTC | Exp 454: VPRM Arithmetic Rule Verifier — rule-base | OK | 81 passed in 53.52s |
| 2026-04-18 18:36 UTC | Exp 455: Think Probe v2 with Partial Verdicts — RE | OK | 81 passed in 38.29s |
| 2026-04-18 18:55 UTC | Exp 456: Constraint Addition from Memory — Tier 1 self-learning relay | OK | 27 tests pass; session1_fp_rate=1.0→session2_fp_rate=0.0; honest_verdict=improvement |
| 2026-04-18 18:59 UTC | Exp 456: Tier 1 Constraint Addition from Memory —  | OK | 81 passed in 47.40s |
| 2026-04-18 19:27 UTC | Exp 457: LSEBMCL Cross-Session EBM Replay — EBM wa | OK | 81 passed in 77.73s (0:01:17) |
| 2026-04-18 20:06 UTC | Exp 458: EBM-CoT Latent Thought Calibration — EORM | OK | 81 passed in 123.49s (0:02:03) |
| 2026-04-18 20:41 UTC | Exp 459: KAEM Large-Variable Crossover Benchmark — | OK | 81 passed in 55.58s |
| 2026-04-18 20:56 UTC | Exp 460: AMD XDNA IRON NPU Unblock — pip install m | OK | 81 passed in 46.05s |
| 2026-04-18 21:12 UTC | Exp 461: Milestone 2026.04.34 Retrospective — did  | OK | 81 passed in 55.09s |
| 2026-04-18 21:43 UTC | Plan milestone 2026.04.35 | OK | 12 tasks proposed |
| 2026-04-18 21:48 UTC | Milestone 2026.04.35 activated | OK | 12 tasks queued |
| 2026-04-18 22:02 UTC | Exp 462: DeliverableGuard + DualGPURunner — close  | OK | 81 passed in 25.38s |
| 2026-04-18 22:19 UTC | Exp 463: Conductor Session Health Check — zombie k | OK | 81 passed in 25.52s |
| 2026-04-18 22:43 UTC | Exp 464: Live Precision 100q — RETRO-033 closure w | OK | 81 passed in 34.53s |
| 2026-04-18 22:56 UTC | Exp 465: ThinkProbeV2 Live GPU Execution — RETRO-0 | OK | 81 passed in 23.40s |
| 2026-04-18 23:14 UTC | Exp 466: EBM-CoT Calibration v3 — RETRO-034 closur | OK | 81 passed in 19.48s |
| 2026-04-18 23:35 UTC | Exp 467: VeriCoT+VPRM Integrated Live Pipeline 200 | OK | 81 passed in 54.51s |
| 2026-04-18 23:59 UTC | Exp 468: GSM-Symbolic Adversarial Benchmark — Appl | OK | 81 passed in 42.71s |
| 2026-04-19 00:13 UTC | Exp 469: HumanEval Live with CodeExtractor + VeriC | OK | 81 passed in 23.40s |
| 2026-04-19 00:32 UTC | Exp 470: PPSEBM Tier 2 Progressive Constraint Para | OK | 81 passed in 23.66s |
| 2026-04-19 00:54 UTC | Exp 471: KV260 FPGA Bring-Up v2 — sparsified Ising | OK | 81 passed in 17.18s |
| 2026-04-19 01:07 UTC | Exp 472: JEPA Tier 3 Scale + GPU-Accelerated Oscil | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-19 01:19 UTC | Exp 472: JEPA Tier 3 Scale + GPU-Accelerated Oscil | OK | 119 passed in 29.84s |
| 2026-04-19 01:33 UTC | Exp 473: Milestone 2026.04.35 Retrospective — did  | OK | 81 passed in 14.72s |
| 2026-04-19 02:00 UTC | Plan milestone 2026.04.36 | OK | 13 tasks proposed |
| 2026-04-19 02:05 UTC | Milestone 2026.04.36 activated | OK | 13 tasks queued |
| 2026-04-19 02:21 UTC | Exp 474: GPUVRAMGate — mid-session zombie kill bef | OK | 81 passed in 11.05s |
| 2026-04-19 02:33 UTC | Exp 475: Conductor Dedup Check + Partial-Result Ha | OK | 81 passed in 10.02s |
| 2026-04-19 02:49 UTC | Exp 476: Live 100q Precision v4 — RETRO-033 third  | OK | 81 passed in 16.27s |
| 2026-04-19 03:12 UTC | Exp 477: JEPA Quality-Gated Retrain — RETRO-040 fi | OK | 81 passed in 9.92s |
| 2026-04-19 03:27 UTC | Exp 478: Live 200q VeriCoT+VPRM v2 — RETRO-038 sec | OK | 81 passed in 7.74s |
| 2026-04-19 03:42 UTC | Exp 479: GSM-Symbolic Adversarial Benchmark live — | OK | 81 passed in 17.71s |
| 2026-04-19 04:00 UTC | Exp 480: Harness DualGPURunner Enforcement — wire  | OK | 81 passed in 13.43s |
| 2026-04-19 04:14 UTC | Exp 481: Inference Batching Enforcement — BatchedI | OK | 81 passed in 3.10s |
| 2026-04-19 04:31 UTC | Exp 482: ThinkProbeV2 Live GPU v3 — RETRO-036/042  | OK | 81 passed in 3.27s |
| 2026-04-19 04:40 UTC | Exp 483: KAEM Profile at n_vars>200 — RETRO-031 cl | OK | 81 passed in 3.30s |
| 2026-04-19 04:52 UTC | Exp 484: Neural Uncertainty Principle Probe — hall | OK | 81 passed in 3.27s |
| 2026-04-19 05:03 UTC | Exp 485: PPSEBM Real-Data Validation — RETRO-043,  | OK | 81 passed in 3.37s |
| 2026-04-19 05:17 UTC | Exp 486: Milestone 2026.04.36 Retrospective — did  | OK | 81 passed in 5.36s |
| 2026-04-19 05:45 UTC | Plan milestone 2026.04.37 | OK | 13 tasks proposed |
| 2026-04-19 05:50 UTC | Milestone 2026.04.37 activated | OK | 13 tasks queued |
| 2026-04-19 06:01 UTC | Exp 487: GPUVRAMGateV2 — kill zombies BEFORE check | OK | 81 passed in 4.70s |
| 2026-04-19 06:13 UTC | Exp 488: Live 100q Precision v5 — RETRO-033 fifth  | OK | 81 passed in 4.85s |
| 2026-04-19 06:25 UTC | Exp 489: Live 200q VeriCoT+VPRM v3 — RETRO-038 thi | OK | 81 passed in 4.93s |
| 2026-04-19 06:38 UTC | Exp 490: GSM-Symbolic Adversarial v3 — RETRO-039 t | OK | 81 passed in 3.96s |
| 2026-04-19 07:03 UTC | Exp 491: JEPA Curriculum Diagnostic — why did qual | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-19 08:01 UTC | Exp 491: JEPA Curriculum Diagnostic — why did qual | OK | 110 passed in 24.30s |
| 2026-04-19 08:39 UTC | Exp 492: JEPA Curriculum Retrain v3 — ordered high | OK | 81 passed in 3.23s |
| 2026-04-19 08:47 UTC | Exp 493: Batching Pre-commit Hook — RETRO-045 CLOSED (hook installed, 10 tests pass) | OK | 10 passed in 8.88s |
| 2026-04-19 08:49 UTC | Exp 493: Batching Enforcement Pre-Commit Hook — RE | OK | 81 passed in 3.12s |
| 2026-04-19 08:58 UTC | Exp 494: GPU Thermal Gate — RETRO-046 third attemp | OK | 81 passed in 3.16s |
| 2026-04-19 09:11 UTC | Exp 495: DualGPU Harness Enforcement v2 — patch 53 | OK | 81 passed in 3.19s |
| 2026-04-19 09:24 UTC | Exp 496: NUP Probe v2 — Bayesian Semantic Entropy  | OK | 81 passed in 3.12s |
| 2026-04-19 09:35 UTC | Exp 497: SuRe Surprise-Driven EBM Replay — Tier 2  | OK | 81 passed in 3.09s |

| 2026-04-19 09:43 UTC | Exp 498: KAEM Extended Profile n=5000 — RETRO-031 CLOSED (fpga_path_recommended, no crossover at n=5000) | OK | 11 passed in 3.43s |
| 2026-04-19 09:45 UTC | Exp 498: KAEM Extended Profile n=5000 — RETRO-031  | OK | 81 passed in 3.06s |
| 2026-04-19 09:57 UTC | Exp 499: Milestone 2026.04.37 Retrospective — did  | OK | 81 passed in 5.10s |
| 2026-04-19 10:29 UTC | Plan milestone 2026.04.38 | OK | 13 tasks proposed |
| 2026-04-19 10:34 UTC | Milestone 2026.04.38 activated | OK | 13 tasks queued |
| 2026-04-19 10:46 UTC | Exp 500: Gemma4 INT4 Quantization — RETRO-048 root | OK | 81 passed in 5.12s |
| 2026-04-19 10:56 UTC | Exp 501: Conductor CPU Routing + VRAM Budget Ledge | OK | 81 passed in 4.00s |
| 2026-04-19 12:07 UTC | Exp 319: Operational retrospective for milestone 2026.04.23 | OK | operational_retro_2026_04_23.json written; n=17 experiments, total=691 min, top bottleneck=Exp 308 (138.0 min); RETRO-001/002 carried forward; NEW-001/002 added; estimated next-milestone speedup ~15% |
| 2026-04-19 12:27 UTC | Exp 502: Live 100q Precision v6 — RETRO-033 sixth | OK | Deliverable already exists in repo |
| 2026-04-19 12:38 UTC | Exp 503: Live 200q VeriCoT+VPRM v4 — RETRO-038 fou | OK | 98 passed in 12.86s |
| 2026-04-19 12:54 UTC | Exp 504: GSM-Symbolic Adversarial v4 — RETRO-039 f | OK | 81 passed in 9.10s |
| 2026-04-19 13:32 UTC | Exp 505: Retroactive DualGPU Harness Sweep — patch | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-19 13:37 UTC | Exp 505: Retroactive DualGPU Harness Sweep — patch | OK | Deliverable already exists in repo |
| 2026-04-19 13:47 UTC | Exp 506: Semantic Energy Tier 0d — Boltzmann-clust | OK | 91 passed in 9.96s |
| 2026-04-19 14:01 UTC | Exp 507: NUP Probe v3 — RETRO-049 fix with CLAP cr | OK | 81 passed in 9.39s |
| 2026-04-19 14:14 UTC | Exp 508: KAEM Distribution Family — RETRO-031 new  | OK | 81 passed in 7.72s |
| 2026-04-19 14:26 UTC | Exp 509: PPSEBM Energy-Magnitude Replay — RETRO-05 | OK | 81 passed, 2 warnings in 2.76s |
| 2026-04-19 14:39 UTC | Exp 510: JEPA Live Retraining v4 — FR-11 Tier 3 se | OK | 81 passed, 2 warnings in 3.06s |
| 2026-04-19 14:50 UTC | Exp 511: AMD XDNA NPU NUP Probe Inference — first  | OK | 81 passed, 2 warnings in 3.38s |
| 2026-04-19 15:03 UTC | Exp 512: Milestone 2026.04.38 Retrospective | OK | 81 passed, 2 warnings in 2.86s |
| 2026-04-19 15:37 UTC | Plan milestone 2026.04.39 | OK | 12 tasks proposed |
| 2026-04-19 15:42 UTC | Milestone 2026.04.39 activated | OK | 12 tasks queued |
| 2026-04-19 15:55 UTC | Exp 513: JITVRAMCheck + Sequential Model Loading — | OK | 81 passed, 2 warnings in 2.82s |
| 2026-04-19 16:19 UTC | Exp 514: Live 100q Precision v7 — RETRO-033 sevent | OK | 81 passed, 2 warnings in 4.16s |
| 2026-04-19 16:30 UTC | Exp 515: Live 200q VeriCoT+VPRM v5 — RETRO-038 fif | OK | 81 passed, 2 warnings in 4.67s |
| 2026-04-19 16:42 UTC | Exp 525: Expanded GPU Reaper — close RETRO-033 roo | OK | 81 passed, 2 warnings in 4.61s |
| 2026-04-19 17:20 UTC | Exp 516: GSM-Symbolic Adversarial v5 — RETRO-039 f | OK | 81 passed, 2 warnings in 2.91s |
| 2026-04-19 17:31 UTC | Exp 517: Controlled DualGPU Parallel Execution Tes | OK | 81 passed, 2 warnings in 2.70s |
| 2026-04-19 17:44 UTC | Exp 518: Top-20 Legacy BatchedInferenceRunner Migr | OK | 81 passed, 2 warnings in 2.76s |
| 2026-04-19 17:57 UTC | Exp 519: CIKANEnergy — Constraint-Informed KAN spl | OK | 81 passed, 2 warnings in 2.73s |
| 2026-04-19 18:08 UTC | Exp 520: LeWorldModel-JEPA Stable Two-Term Trainin | OK | 81 passed, 2 warnings in 2.72s |
| 2026-04-19 18:22 UTC | Exp 521: Hallucination Basin Detector — latent-spa | OK | 81 passed, 2 warnings in 2.80s |
| 2026-04-19 18:36 UTC | Exp 522: JEPA Live Retrain v6 — FR-11 mandatory, L | OK | 81 passed, 2 warnings in 2.69s |
| 2026-04-19 18:47 UTC | Exp 523: NUP Probe v4 — Contrastive Training Objec | OK | 81 passed, 2 warnings in 2.70s |
| 2026-04-19 18:58 UTC | Exp 524: Milestone 2026.04.39 Retrospective | OK | 81 passed, 2 warnings in 6.91s |
| 2026-04-19 19:27 UTC | Plan milestone 2026.04.40 | OK | 11 tasks proposed |
| 2026-04-19 19:32 UTC | Milestone 2026.04.40 activated | OK | 11 tasks queued |
| 2026-04-19 19:43 UTC | Exp 526: env_autofix RETRO-053 Fix — RETRO-053 CLOSED | OK | 34 passed; retro_053_resolved=true |
| 2026-04-19 19:44 UTC | Exp 526: env_autofix CARNOT_FORCE_LIVE='0' Fix — R | OK | 115 passed, 2 warnings in 6.36s |
| 2026-04-19 20:43 UTC | Exp 527: Live 100q Precision v8 — RETRO-033 eighth | OK | 81 passed, 2 warnings in 7.01s |
| 2026-04-19 20:55 UTC | Exp 528: Live 200q VeriCoT+VPRM v7 — RETRO-038 sev | OK | 81 passed, 2 warnings in 7.17s |
| 2026-04-19 21:06 UTC | Exp 529: GPU1 Explicit Routing Fix — RETRO-052 clo | OK | 81 passed, 2 warnings in 2.73s |
| 2026-04-19 21:20 UTC | Exp 530: Wire NUP Probe v4 + Hallucination Basin D | OK | 146 passed, 2 warnings in 7.91s |
| 2026-04-19 21:31 UTC | Exp 531: EORM as Test-Time PRM — Adaptive Rectific | OK | 81 passed, 2 warnings in 2.77s |
| 2026-04-19 21:47 UTC | Exp 532: LowRankKAEMEnergy — SVD projection of log | OK | 81 passed, 2 warnings in 2.79s |
| 2026-04-19 21:57 UTC | Exp 533: COLD Decoding Energy Guidance — token-lev | OK | 81 passed, 2 warnings in 2.77s |
| 2026-04-19 22:13 UTC | Exp 534: PottsMachineVerifier — multi-value constr | OK | 81 passed, 2 warnings in 2.74s |
| 2026-04-19 22:21 UTC | Exp 535: JEPA Live Retrain v7 — FR-11 mandatory, r | OK | 81 passed, 2 warnings in 2.69s |
| 2026-04-19 22:33 UTC | Exp 536: Milestone 2026.04.40 Retrospective | OK | 81 passed, 2 warnings in 5.31s |
| 2026-04-19 22:57 UTC | Plan milestone 2026.04.41 | OK | 12 tasks proposed |
| 2026-04-19 23:02 UTC | Milestone 2026.04.41 activated | OK | 12 tasks queued |
| 2026-04-19 23:12 UTC | Exp 537: ExperimentTemplate.teardown() + GPU Zombi | OK | 81 passed, 2 warnings in 2.73s |
| 2026-04-19 23:35 UTC | Exp 538: Live 25q Precision v9 — RETRO-033 Attempt | OK | 81 passed, 2 warnings in 2.75s |
| 2026-04-19 23:50 UTC | Exp 539: Live 100q VeriCoT+VPRM v8 — RETRO-038 Att | OK | 81 passed, 2 warnings in 2.82s |
| 2026-04-20 00:02 UTC | Exp 540: GRPO Contrastive EORM Retrain — arXiv 250 | OK | 81 passed, 2 warnings in 2.86s |
| 2026-04-20 00:12 UTC | Exp 541: ConstraintAdditionFromMemory Live Wire-In | FAIL | Claude Code error: Error: Reached max turns (50) |
| 2026-04-20 00:17 UTC | Exp 541: ConstraintAdditionFromMemory Live Wire-In | OK | Deliverable already exists in repo |
| 2026-04-20 00:24 UTC | Exp 542: FOVER Corpus Expansion — FR-11 upstream,  | OK | 151 passed, 2 warnings in 3.15s |
| 2026-04-20 00:43 UTC | Exp 543: JEPA v8 Live Retrain — FR-11 mandatory, e | OK | 81 passed, 2 warnings in 4.92s |
| 2026-04-20 01:26 UTC | Exp 544: LowRankKAEMEnergy Cascade Integration — 2 | OK | 145 passed, 2 warnings in 5.96s |
| 2026-04-20 01:38 UTC | Exp 545: InternalStateProbe — arXiv 2511.06209, 81 | OK | 81 passed, 2 warnings in 2.74s |
| 2026-04-20 01:51 UTC | Exp 546: AutoRefine Constraint Template Distillati | OK | 81 passed, 2 warnings in 2.70s |
| 2026-04-20 02:05 UTC | Exp 547: Legacy Modernization Sprint — BatchedInfe | OK | 81 passed, 2 warnings in 3.92s |
| 2026-04-20 02:19 UTC | Exp 548: Milestone 2026.04.41 Retrospective | OK | 81 passed, 2 warnings in 5.86s |
| 2026-04-20 02:49 UTC | Plan milestone 2026.04.42 | OK | 14 tasks proposed |
| 2026-04-20 02:54 UTC | Milestone 2026.04.42 activated | OK | 14 tasks queued |
| 2026-04-20 03:07 UTC | Exp 549: Conductor Exclusion Manifest + nvidia-smi | OK | 81 passed, 2 warnings in 5.60s |
| 2026-04-20 03:25 UTC | Exp 550: BatchedInferenceRunner Real Migration — E | OK | 81 passed, 2 warnings in 9.30s |
| 2026-04-20 03:36 UTC | Exp 551: Live 50q Data Collection A — GSM8K indice | OK | 81 passed, 2 warnings in 8.23s |
| 2026-04-20 03:55 UTC | Exp 552: Live 50q Data Collection B — GSM8K indice | OK | 81 passed, 2 warnings in 5.66s |
| 2026-04-20 04:07 UTC | Exp 553: FOVER Corpus v2 — Merge, Diversity Audit, | OK | 81 passed, 2 warnings in 5.42s |
| 2026-04-20 04:19 UTC | Exp 554: VeriCoT+VPRM Extraction Diagnostic on Exp | OK | 81 passed, 2 warnings in 5.31s |
| 2026-04-20 04:33 UTC | Exp 555: Confidence-Weighted Constraint Filtering  | OK | 81 passed, 2 warnings in 5.11s |
| 2026-04-20 04:53 UTC | Exp 556: EORM GRPO Retrain on 100+ Real Pairs — RE | OK | 81 passed, 2 warnings in 4.81s |
| 2026-04-20 05:06 UTC | Exp 557: JEPA v9 Retrain — Diverse 100+ Corpus, Le | OK | 81 passed, 2 warnings in 4.83s |
| 2026-04-20 05:16 UTC | Exp 558: InternalStateProbe Real-Data Training — a | OK | 81 passed, 2 warnings in 4.98s |
| 2026-04-20 06:21 UTC | Exp 559: LowRankKAEM Calibration Layer — RETRO-057 | FAIL | Claude Code error: Wall-clock timeout after 3603s. Last output:  |
| 2026-04-20 06:33 UTC | Exp 559: LowRankKAEM Calibration Layer — RETRO-057 | OK | 100 passed, 2 warnings in 13.18s |
| 2026-04-20 06:46 UTC | Exp 560: LatentCoTEBMCalibrator — Step-Level Energ | OK | 81 passed, 2 warnings in 3.22s |
| 2026-04-20 06:59 UTC | Exp 561: Tier 1 Self-Learning Relay on Real Data — | OK | 81 passed, 2 warnings in 2.78s |
| 2026-04-20 07:13 UTC | Exp 562: Milestone 2026.04.42 Retrospective | OK | 81 passed, 2 warnings in 2.82s |
| 2026-04-20 07:42 UTC | Plan milestone 2026.04.43 | OK | 12 tasks proposed |
| 2026-04-20 07:47 UTC | Milestone 2026.04.43 activated | OK | 12 tasks queued |
| 2026-04-20 07:58 UTC | Exp 563: Live 50q Data Collection A v2 — RETRO-062 | OK | 81 passed, 2 warnings in 2.80s |
| 2026-04-20 08:09 UTC | Exp 564: CoACEExtractor — Code-Assisted Constraint | OK | 81 passed, 2 warnings in 2.76s |
| 2026-04-20 08:21 UTC | Exp 565: Live CoACEExtractor Diagnostic — TP/FP on | OK | 81 passed, 2 warnings in 2.79s |
| 2026-04-20 08:32 UTC | Exp 566: JEPAPUREMinForm — Min-Form PRM Objective  | OK | 81 passed, 2 warnings in 2.74s |
| 2026-04-20 08:45 UTC | Exp 567: JEPA v10 Retrain — PURE MinForm on 132-pa | OK | 81 passed, 2 warnings in 3.04s |
| 2026-04-20 08:56 UTC | Exp 568: KV260 FPGA Bring-Up v2 — Board Arrived 20 | OK | 81 passed, 2 warnings in 2.74s |
| 2026-04-20 09:09 UTC | Exp 569: Live Verify-Repair with CoACEExtractor —  | OK | 81 passed, 2 warnings in 2.78s |
| 2026-04-20 09:23 UTC | Exp 570: FR-11 Tier 1 Self-Learning Relay — Real C | OK | 81 passed, 2 warnings in 2.74s |
| 2026-04-20 09:36 UTC | Exp 571: HalluField Tier 0e — Thermodynamic Energy | OK | 145 passed, 2 warnings in 3.14s |
| 2026-04-20 09:54 UTC | Exp 572: PRA EBM Beam Search — EORM as Step-Level  | OK | 81 passed, 2 warnings in 2.73s |
| 2026-04-20 10:10 UTC | Exp 573: Energy-per-Token EORM Hardware Calibratio | OK | 81 passed, 2 warnings in 2.80s |
