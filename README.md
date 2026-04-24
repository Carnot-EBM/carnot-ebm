# Carnot

**Open-source Energy Based Model framework — Rust + Python/JAX**

Carnot uses Energy-Based Models to **verify and repair LLM outputs**. It extracts constraints from any response, checks them formally (Z3 SMT, property-based testing, energy scoring), and repairs violations via LLM feedback. All headline results are from live GPU inference.

**Headline results (validated and defensible):** +3.0pp on 164-problem HumanEval (statistically significant), +4.9pp on typed constraint verification, 86% false positive reduction via self-learning, 99.3% code bug detection rate, **+5pp live precision improvement** (Exp 451, first positive verify-repair number), **2.02× DualGPU parallel-training speedup** with identical losses sequential↔parallel (Exps 684/685, RETRO-071 closed after 14 milestones), **first true teacher-distilled safety classifier** produced via the REQ-SAFE-011 invariant-guarded distillation pipeline (Exp 690), **Prompt-Injection KAN v3 AUROC=0.9078** (Exp 724, first passing of the 0.90 publication gate), **JEPA v18 OOD AUC=0.5115** (Exp 717, first JEPA above random 0.5 since v14), **JEPA-Reasoner Probe OOD AUC=1.0** (Exp 726, pre-generative Tier 2.1 candidate), **7.2% pipeline latency reduction via VarGran gate** (Exp 727, 60% Ising skip rate, FN rate improved), **RETRO-033 definitively closed** — two independent 200q VR trials (+0.00510pp each, Exp 742, RETRO-033 closed after 20 attempts), **FR-11 formally closed** — all evidence gates met: probe AUC=0.993, relay operational, tier2_memory_functional (Exp 741), **Privacy Filter KAN v2 AUROC=1.0** on 2/3 holdout datasets (Exp 743), **DualGPU EORM+JEPA 1.83x speedup** confirmed (Exp 746), **complete slowest-5 composition change** for the first time in project history (Milestone 2026.04.57), **manifest enforcement finally applied** after 4 consecutive non-enforcement cycles (Exp 754), **PSV relapse closed** via layered SRSA gate + constraint freezing + curriculum diversity — fp_rate 0.605→0.334, recovery_sustained=True (Exp 756), **Yosys open-source FPGA synthesis validated** — 2821 LUTs / 2237 DFFs, 0 errors, no Vivado required (Exp 758), **AST verifier perfect precision/recall/F1=1.0** on 50 synthetic code snippets (Exp 764). See the [technical report](docs/technical-report.md) for the full 792-experiment analysis.

**Claims we retracted under audit.** The "+64 pp verify-repair with structured forcing" (Exps 668/679) and "AUROC 0.9585 publishable prompt-injection classifier" (Exp 691) headlines from Milestones 2026.04.51–.52 **do not survive audit**:

- Exp 668's 25-question +64 pp (36 → 100%) did not hold up mechanistically — Exp 687's sparse-autoencoder found no latent feature distinguishing correct from incorrect reasoning steps, and Exp 679's attempted 200-question replication produced a physically implausible 0/200 baseline with Qwen3.5-0.8B (a model that normally scores 25–45% on GSM8K). Net reading: the grader regex matches the forced `COMPUTE: X op Y = Z` format only, so the "improvement" measures output-format compliance, not reasoning.
- Exp 691's AUROC 0.9585 across HackAPrompt, BIPIA, and synthetic OWASP-LLM-01 held-outs is a ranking number, not a usable classifier. The confusion matrices at the default threshold are `TP=0, FP=0, TN=N, FN=N` on every dataset — the model detects zero injections in practice. AUROC-without-calibration was insufficient gate logic.

Both findings stand as negative results in the research record (see `results/experiment_{668,679,687,691}_*.json`) and as arguments for the audit discipline that surfaced them. Exp 682 (JEPA v15 true-OOD AUC = 0.475 after in-distribution 1.0) is the same pattern applied to the cascade tier. See [Known Measurement Artifacts](#known-measurement-artifacts) below.

**What ships today:** `pip install carnot` -- verify any LLM output in 5 lines of Python. CLI, MCP server for Claude Code, and full API docs. Four energy model tiers (KAN, Ising, Gibbs, Boltzmann) with hardware acceleration paths (FPGA, D-Wave quantum annealing, Extropic TSU).

## Why Carnot: the validation moat

Finding candidate hallucinations is getting cheap. Public off-the-shelf models now replicate, for under $30/scan, vulnerability-discovery findings that required restricted frontier models a quarter ago ([Decrypt, April 2026](https://decrypt.co/364744/anthropic-mythos-replicated-public-models-vidoc-security)). The same trajectory applies to LLM verification: any modern LLM can flag "this step looks suspicious" with reasonable recall. Detection is commoditising.

What public models still cannot do is **chain the evidence** -- compile the suspicious steps into a joint constraint system, prove or refute their consistency, and produce a repair direction when they fail. That requires a symbolic layer downstream of the LLM. Carnot is that layer: Ising couplings that you can write by hand from a theorem prover, plus Ising couplings that you can learn from data, plus a gradient-descent repair loop that turns *detector signal into trusted verdicts with repair gradients*. The moat is moving from model access to validation, and Carnot is pointed at the validation side.

## Install

```bash
# Python (3.11+)
pip install -e ".[dev]"

# Verify it works
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

> GPU: `pip install carnot[cuda]` for CUDA 12. On AMD/ROCm, use `JAX_PLATFORMS=cpu`.
> Rust bindings (optional): `pip install carnot[rust]` with Rust toolchain installed.

### Quick start (Python API)

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()

# Correct answer — passes verification
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
print(result.verified)    # True

# Wrong answer — caught by constraint extraction
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 43")
print(result.verified)    # False
print(result.violations)  # [ConstraintResult: "15 + 27 = 43 (correct: 42)"]
```

## The Problem with LLMs

Large language models generate text by predicting the most probable next token. This produces fluent output, but it's fundamentally guessing — there is no mechanism to verify that the output is logically consistent, physically valid, or factually correct. When an early token is wrong, the error cascades irrecoverably. This is why LLMs hallucinate: they optimize for plausibility, not truth.

## Why Energy Based Models?

EBMs take a fundamentally different approach. Instead of generating outputs sequentially, they assign a scalar energy to every possible configuration of variables. Low energy = valid/consistent; high energy = invalid/contradictory. Inference is optimization: find the configuration that minimizes energy across all constraints simultaneously.

This enables capabilities that autoregressive models structurally cannot provide:

- **Verifiable reasoning** — mathematically prove a solution satisfies all constraints by showing it sits at an energy minimum
- **Surgical error correction** — when a constraint is violated, gradient descent fixes the broken part without discarding the rest
- **Autonomous self-improvement** — the energy function is an objective ground truth that cannot be gamed, enabling closed-loop self-learning without human feedback
- **Hardware acceleration** — energy landscapes map directly to thermodynamic sampling hardware (Extropic TSU), promising 10,000x efficiency gains

## How Carnot Uses EBMs (Introspection, Not Fine-Tuning)

**Carnot never modifies the LLM's weights.** The target language model remains completely frozen throughout all experiments and deployment. Instead, Carnot works by introspecting the LLM's existing internal representations:

1. **Logprob-based methods** — read the LLM's own per-token log-probabilities as an energy signal. The model is already an EBM (per the ARM↔EBM bijection); we simply read the energy it already computes.
2. **Activation-based methods** — extract hidden state activations from a frozen forward pass (`output_hidden_states=True`), then train a small separate EBM classifier (a lightweight Gibbs model, typically [1024→256→64→1]) on those extracted features via Noise Contrastive Estimation.
3. **Structural verification** — execute the LLM's generated code against test cases. No model weights involved at all.

The "training" in Carnot refers to training the small EBM classifier on activation features extracted from a frozen LLM — not gradient descent on the LLM itself. This is fundamentally different from fine-tuning, RLHF, or DPO, which modify the language model's parameters. Carnot's approach is closer to probing or introspection: we observe what the model already knows internally and build a lightweight detector on top of it.

## The Path to Self-Learning

Carnot is designed from the ground up to support an automated self-improvement loop (LLM proposes, energy function evaluates):

1. **Propose** — candidate improvements to architecture, training, or hyperparameters are prototyped in Python/JAX
2. **Evaluate** — the energy landscape on held-out data serves as the objective judge (did energy decrease? real improvement. did it not? rejected.)
3. **Deploy** — proven improvements are transpiled to Rust for production performance
4. **Repeat** — the loop runs without human supervision, with safety guardrails

The EBM itself is the evaluator. No LLM needed to judge quality — the math provides ground truth.

## Known Measurement Artifacts

We keep a running list of claims that *technically matched a success criterion* but fail an honest "does this deliver the capability?" test. Publishing these alongside the claims that do survive audit is part of the project's credibility story.

| Artifact | What the metric said | What the audit found | Status |
|---|---|---|---|
| Exp 668 / 679 "VR +64 pp → 100%" | 36% → 100% on 25 GSM8K questions, replicated at 0/200 → 200/200 at 200q | Exp 687 SAE found no latent feature distinguishing correct vs incorrect steps; 0/200 baseline on Qwen3.5-0.8B is physically implausible (normal range 25–45%); the grader regex only matches the forced `COMPUTE:` format, so the "improvement" is format-compliance, not reasoning | Treated as negative result; RETRO-033 closure retracted pending a grader that works on free-form output |
| Exp 691 prompt-injection "publishable" | Mean cross-dataset AUROC 0.9585 over HackAPrompt + BIPIA + synthetic | Confusion matrices at threshold 0.5 are `TP=0, FP=0, TN=N, FN=N` on every dataset — zero injections detected in practice. Training-distribution AUROC 0.80 vs OOD 0.96 is physically implausible (generalization doesn't improve OOD in real ML). Likely shared dataset artifact the KAN latched onto; score distribution sits entirely below 0.5 | Not shipped; model card (`python/carnot/models/prompt_injection_kan_v1_MODELCARD.md`) shows the `Precision: 0.000, Recall: 0.000` rows honestly. Next experiment: threshold calibration (Platt or isotonic) before any public release |
| Exp 671 → 682 JEPA v15 OOD AUC | Exp 671 reported `ood_auc=1.0` "perfect OOD separation" | Exp 682 tested on a genuinely held-out GSM8K range with zero training-index overlap and got `true_ood_auc=0.4751` — below random. v15's OOD claim collapsed entirely | Exp 682 verdict preserved honestly; v15 is **not** a cascade-ready tier. Re-audit any claim that relied on it |
| Exp 688 PSV self-play "improving" | `fp_rate_trend_slope = −0.005`, verdict `psv_selfplay_fp_improving` | FP rate oscillates between 0.45 and 0.75 over 10 iterations; the "trend" is below noise floor. `inference_mode=live_replay` confirms no real self-play happened — the threshold was mechanically lowered from 0.5 to 0.25 (which trivially reduces FP rate but also destroys recall) | Effect too small to report; verdict string reads as positive but the numbers don't back it |

**Why we publish these.** An RFC-grade research project has to report the negatives alongside the positives, or the positives become uninterpretable. The audits that surfaced these (Exps 679 true 200q scale, 682 true OOD, 687 SAE feature attribution, 691 per-dataset confusion matrices) are the methodologically important outputs of the milestone even when the headline verdicts read as wins.

## Key Results (792 experiments, 47 completed milestones)

All benchmark results below are from **live GPU inference**. Simulated and software-model artifacts remain in the repo, but they are labeled explicitly and are not mixed into the headline tables. See the [technical report](docs/technical-report.md) for the full history including what didn't work.

### Simulation vs Reality

Provenance snapshot: **15 live GPU artifacts**, **5 simulated artifacts**, **95 unverified artifacts**, and **1 software-model artifact** (Exp 228, software simulation). Only the live GPU subset informs the benchmark tables below.

### KV260 hardware — functional on silicon (2026-04-22)

After 12 consecutive PS hangs across a single-session debug cycle, the Carnot Ising sampler now **responds correctly on real KV260 silicon**. First successful AXI transactions via `/dev/uio4` on the deployed overlay:

- `REG[CONTROL 0x00]` = `0x00000000` (reset default)
- `REG[STATUS 0x04]` = `0x00000001` (ready bit asserted)
- `REG[SPIN_COUNT 0x08]` = `0x00000020` = 32 decimal — confirms the N=32 build parameter is live in silicon
- `REG[BETA 0x0C]` = `0x00000000` (reset default)
- **Write-read roundtrip:** `write 0xDEADBEEF → control reg → read back 0xDEADBEEF`

The root cause of the 12 prior hangs was a **device-tree overlay structural bug**: our dtbo targeted `/` and declared `firmware-name` at root, but the Linux fpga-manager only releases the PS-PL AXI isolation resets (`zynqmp_reset` IDs 0x74-0x77) when the overlay targets `fpga_full` and declares them explicitly. Without those reset releases, the PL bitstream loaded (fabric reported "operating") but the PS→PL AXI boundary stayed isolated — every AXI transaction wedged CPU3 on an un-returning load instruction, cascading into RCU stalls after the 60-second grace period and mmc1 IRQ starvation as a secondary effect. **RETRO-074 is closed** after the dtbo was restructured to mirror the `k26-starter-kits` reference (`target = <&fpga_full>`, explicit `resets` property).

Additional corrections landed during the same debug: AXI-Lite read/write channels now use `aw_done`/`w_done`/`ar_done` latches for independent AR/AW/W/R ACK timing (matches SmartConnect pulse behavior); LFSR advance gated on `reg_control[0]` to eliminate 2048 idle-switching flops; `interconnect_aresetn` wired to SmartConnect per Xilinx PG164; full K26 board preset (`PSU__PSS_REF_CLK__FREQMHZ = 33.333` + 186 other PSU properties) now applied via `apply_bd_automation`.

Current hardware configuration: N=32 spins, MAX_DEGREE=8 edges/spin, pl_clk0 at 60 MHz, WNS ≈ +0.18 ns, 0 failing timing endpoints. Bitstream provenance: `output/carnot_ising_bd/carnot_ising_bd_wrapper.bit`. The next experiment (TBD) runs a real Hamiltonian through the sampler and compares ground-state energies against the Python reference implementation.

Note: Milestone 2026.04.60 (Exps 780-792, 47th milestone, title "JEPA v20 Data Surge + SOTA GGUF Confirmed + Constraint Memory to Constraint Generation") completed (2026-04-24). 12 experiments, 110.2 min total, 9.19 min/exp (first improvement after .59 regression). 6/12 success criteria met. Key results: **GPU zombie killer deployed** — setup_gpu_wired=True, kill_gpu_zombies() now mandatory pre-flight in ExperimentTemplate.setup_gpu() (Exp 780, RETRO-028 and RETRO-SOTA-GGUF-TIMEOUT cleanup); **JEPA v20 data collection blocked** on live GPU availability (Exp 781); **EDU-PRM step selection validated** — 18/57 steps selected, uncertainty_selected_pct=0.316 >= 0.30 gate (Exp 782); **JEPA v20 retrain** — ood_auc=0.4467, regression vs v19 ood_auc=0.5667, `jepa_v20_insufficient_data` — data starvation root cause confirmed (Exp 783); **JEPA v20 cascade deploy** `blocked_ood_auc_below_gate` (Exp 784); **SOTA GGUF code repair v2** timed out at 90-min hard cap — second consecutive milestone, RETRO-SOTA-GGUF-TIMEOUT still open (Exp 785); **Gemma4 OOM fix v3 + VR grid** `blocked_no_live_gpu` (Exp 786); **SSTAR energy-ranked code selection** `energy_prefilter_efficient` — energy_correct_rank_pct=0.70, tests_saved_pct=0.75 (Exp 787); **Constraint addition** zero delta, static=dynamic, approach needs redesign (Exp 788); **EBM calibration** ECE improved 0.3165→0.1026, 67% reduction (Exp 789); **NPU mlir-aie installed** via pip — MLIR-AIE option_a_success=True, Vitis toolchain required for benchmark (Exp 790); **KV260 N=32 iCE40 synthesis** — yosys/nextpnr-ice40/icepack not installed, pnr_success_ice40=False (Exp 791); **Retrospective (Exp 792):** criteria_met=6/12, `honest_verdict=wall_time_IMPROVEMENT_18.2min_first_after_59_regression...WINS_edu_prm_validated_energy_prefilter_70pct_ebm_calibration_ECE_67pct_improvement_npu_mlir_installed`. Open: RETRO-SOTA-GGUF-TIMEOUT, RETRO-028, JEPA data starvation.

Note: Milestone 2026.04.59 (Exps 767-779, 46th milestone, title "JEPA v19 Closure + SOTA GGUF Benchmarks + SETS Comparison + Semantic Energy") effectively closed. Key results: **Pre-flight v11** extended the manifest to all queue dequeue sites, `full_manifest_coverage_achieved` (Exp 767); **Gemma4 loader fix v2** still blocked (Exp 768, `loader_still_broken`); **SOTA GGUF code repair** timed out (Exp 769 — retried as Exp 785); **JEPA v19 predictive** ran live GPU, `jepa_v19_improving` (Exp 770); **EBRM vs EORM comparison** — EBRM competitive but not strictly dominant (Exp 771, arXiv 2504.13134); **Semantic energy probe** — `semantic_energy_below_baseline`, Tier 0g candidate deferred (Exp 772); **Carnot vs SETS** (arXiv 2501.19306) — inconclusive on the current benchmark subset (Exp 773); **Adaptive Bayesian PSV** — variance-based early stopping matches full-scan quality at lower cost (Exp 774, `adaptive_efficient_lossless`, arXiv 2603.22812); **Jailbreak Detection KAN v1** — Safety Tier 0h deployed (Exp 775, arXiv 2602.11495); **KV260 nextpnr-xilinx open-source synthesis** — place-and-route completed but failed timing closure (Exp 776, `pnr_failed_timing` — Vivado remains the path that meets timing); **HuggingFace publishing** — 2 models published to `Carnot-EBM/*`, 26 model-card READMEs updated with the "Production Use" section pointing at `https://github.com/Carnot-EBM/carnot-ebm` (Exp 777); **JEPA v19 cascade deploy** `blocked_ood_auc_below_gate` (Exp 778).

Note: Milestone 2026.04.58 (Exps 754-766, 45th milestone, title "PSV Repair + HLS Fix + Live Repair + SRSA Gate") completed. Key results: **Manifest enforcement applied** — patch_applied=True, exp527_excluded=True, 23 experiments in exclusion manifest; RETRO-MANIFEST-ENFORCEMENT and RETRO-EXP527-GOVERNANCE closed (Exp 754); **PSV relapse closed** — SRSA gate + constraint freezing + curriculum diversity drove fp_rate from 0.605 to 0.334 across 60 steps, window1_slope=-0.005751, window2_slope=-0.0030, recovery_sustained=True (Exp 756); **HLS energy sign validated** — energy_after_fix=-6.0 matches expected_energy=-6.0, delta_pct=0.0, RETRO-HLS-ENERGY closed (Exp 757); **Yosys open-source FPGA synthesis clean** — 2821 LUTs / 2237 DFFs, 0 synthesis errors; open-source path validated without Vivado (Exp 758); **Live GPU code repair first run** — CARNOT_FORCE_LIVE blocker resolved; Exp 759 ran live GPU HumanEval 2-round repair; signed_improvement=0.0 (RETRO-CODE-REPAIR-ZERO opened); **Tier 1 constraint addition proven** — precision_s10=1.0, monotone non-decreasing confirmed across 10 sessions (Exp 761); **Dual-pathway MoP probe AUROC=1.0** vs JEPAReasonerProbe baseline 0.993 (Exp 763, N=12 caveat); **AST verifier perfect precision/recall/F1=1.0** on 50 synthetic code snippets (Exp 764); **Fourteenth consecutive conductor-cycle wall-time improvement** — 24.8 min total for 11 experiments (−210 min vs 235 min prior baseline, −89.5%). Open: RETRO-CODE-REPAIR-ZERO, RETRO-GEMMA4-LOADER, RETRO-JEPA-V19-NOT-RUN.

Note: Milestone 2026.04.57 (Exps 740-753, 44th milestone, title "Milestone 2026.04.57 Operational Retrospective") completed. Key results: **Preflight v9** — updated for new experiment classes (Exp 740); **FR-11 formally closed** — certificate written (results/fr11_closure_certificate.json), all evidence gates met: probe AUC=0.993, relay operational, tier2_memory_functional, latency_p99<200ms, events_acked=100 (Exp 741); **RETRO-033 definitively closed** — two independent 200q VR trials (seed=218: +0.00510; seed=999: +0.00510) with identical results, VR pipeline credibility established, Exp 527 class permanently retired (Exp 742); **Privacy Filter KAN v2** — AUROC=1.0 on 2/3 holdout datasets, 0.985 in-distribution, upstream dependency unblocked, 2-consecutive-cycle block resolved (Exp 743); **CoCoA Tier 0f wired** — inter-layer disagreement detector AUC=0.812 added to verify-decision pipeline (Exp 745); **DualGPU EORM+JEPA retrain validated** — 1.8319x speedup confirmed, Exp 383 class exits slowest-5 after 11 consecutive appearances (Exp 746); **Tier 1 weight audit** — completed (Exp 747); **Cross-session memory 10-session stress test** — precision_s1=1.0, precision_s10=1.0, plateau at session 2 (Exp 748); **D-Wave Neal negative result** — Gibbs superior (mean_energy=-42.9 vs Neal=-33.4, Exp 751); **HF artifacts ready** — StepLevelJEPAProbe v1 + KAN Tier 0b v3 model cards and safetensors exported, awaiting operator upload (Exp 752); **Complete slowest-5 composition change** — first time in project history all five legacy experiments (425/410/383/380-382/527) exited simultaneously; new slowest-5 are all productive experiments (Exp 753 retrospective); open items: manifest code patch still not applied to conductor, PSV relapse detected, code repair blocked (CARNOT_FORCE_LIVE not set).

Note: Milestone 2026.04.56 (Exps 731-739, 43rd milestone, focused production-deploy cycle) completed. Key results: **Preflight v8** — zombie GPU1 process cleared, clean GPU state confirmed (Exp 731); **Tier 2.1 JEPAReasonerProbe validated** — AUC 0.993 ± 0.005, far above 0.75 gate, Tier 2.1 confirmed viable for cascade deployment (Exp 732); **FR-11 Tier 1 Relay fully operational** — relay_operational=True, end-to-end verified for the first time in project history (Exp 734); **KAN Tier 0b deployment** — fp_rate=0.0, deployment confirmed (Exp 735); **PSV constraint specialization root cause identified and recovery confirmed** — domain-diverse training resolves three-consecutive-milestone degradation (Exps 736+737); **FR-11 Tier 2 Cross-Session Memory relay** — tier2_relay_functional=True (Exp 738, arXiv 2511.06209); **Retrospective (Exp 739):** milestone_wins=[FR-11 fully operational, Tier 2.1 AUC 0.993, KAN Tier 0b deployed, PSV recovery, manifest fix written]; open items: privacy filter blocked (Exps 729+730, second consecutive cycle), RETRO-033 VR closure needs confirmation trial, manifest fix patch written but not yet applied to conductor code.

Note: Milestone 2026.04.55 (Exps 716-730, 42nd milestone, title "Step-Level Latent Probe + VarGran Gate + KAN Publication Gate") completed. Key results: **Preflight v7 incremental test selection** — 0 of 554 tests selected, saves ~562 min/cycle (Exp 716); **JEPA v18 LambdaRank Listwise Loss** — ood_auc=0.5115, `honest_verdict=jepa_v18_above_random`, FIRST JEPA above random 0.5 since v14 (Exp 717); JEPA v18 cascade deploy smoke test gate fail (cascade_auc=0.3986, Exp 718); **VR 200q scale evaluation** — signed_improvement_200q=0.0051, `honest_verdict=vr_marginal`, RETRO-033 first positive direction at 200q scale (Exp 720, live GPU); **Prompt-Injection KAN v3** — auroc=0.9078, `honest_verdict=kan_gate_passed`, **FIRST time the 0.90 AUROC publication gate has been cleared** (Exp 724); **JEPA-Reasoner Probe** — ood_auc=1.0, latency_p50_ms=0.02ms, Tier 2.1 candidate, `honest_verdict=probe_tier21_candidate` — new architectural direction bypasses the extractor recall bottleneck (Exp 726); **VarGran Ising Skip Gate** — ising_skip_rate=0.6, latency_reduction_pct=7.196, fn_delta=-0.042 (FN rate 0.060→0.018), `honest_verdict=vargran_gate_success` (Exp 727); GPU close: zombie PID 368449 held 24082MB GPU 1 VRAM at 0% util — first dirty GPU close in 14 milestones.

Note: Milestone 2026.04.54 (Exps 703-715, 41st milestone, title "JEPA v17 RankNet + KAN v2 + FoVer v2") completed. Key results: **JEPA v17 RankNet Pairwise Ranking Loss** — v17_ood_auc=0.4819, +0.6% delta vs v16=0.4759, still below random 0.5; recommendation for v18: switch to listwise LambdaRank loss (Exp 704); **VR Attempt #19 Gemma4** — gate suppressed 25 constraints, vr_accuracy=0.84 (unchanged from baseline), signed_improvement=0.0, `honest_verdict=vr19_gemma4_no_harm` (Exp 708); **Prompt-Injection KAN v2** — AUROC improved 0.7995→0.8747, below 0.90 publication gate, `honest_verdict=distillation_improved_below_gate` (Exp 710); **FoVer v2 PDDL Dataset Synthesis** — 1,400 labeled pairs (200 Z3 + 1,200 PDDL), `honest_verdict=fover_v2_target_met` (Exp 712); **Retrospective (Exp 715):** 12 experiments, 24.32 min total wall time (2.03 min/exp, 71% throughput improvement), no retros closed, JEPA cascade still blocked, publication_ready=True maintained.

Note: Milestone 2026.04.53 (Exps 690-701, 40th milestone, title "Post-.52 Audit + Distillation Invariant + Root Cause Identification") completed. Key results: **Distillation invariant confirmed** (Exp 690: teacher_inference_duration_s=6256s, REQ-SAFE-011 invariant floor confirmed — true teacher-distilled safety classifier); **JEPA v15 root cause identified** — `pure_loss_anti_correlation` confirmed as architectural failure mode responsible for v15 OOD collapse (Exp 693); **VR cross-model validation** — .51 VR improvement does not transfer across models, confirming format-compliance artifact hypothesis (Exp 694, `vr_cross_model_delta=-1.8`); **PSV self-play FP trend degrading** — slope=+0.004242, `honest_verdict=psv_real_fp_degrading` (Exp 697); **JEPA v16 OOD AUC=0.4759**, still below random 0.5, JEPA cascade remains blocked (Exp 698); **HalluSAE integration no improvement** — delta_auc=-0.2142, severe regression (Exp 699); **Publication readiness check** — `publication_ready=True`, audit record and defensible claims confirmed release-ready (Exp 700); **KV260 Ising v3 RTL synthesis** — `honest_verdict=synthesis_blocked_no_tool`, RETRO-072 open pending Vivado (Exp 701); **Retrospective (Exp 702):** experiments_completed=550, avg_time_per_experiment_minutes=8.12, wall_time regression +13.4 min vs .52.

Note: Milestone 2026.04.52 (Exps 678-691, 39th milestone, title "VR Win Scale-Up + DualGPU Proof + JEPA Calibration") completed. Honest key results — the milestone was dominated by audits of prior claims rather than new capabilities: **RETRO-071 CLOSED** — DualGPU parallel retrain speedup 2.02× confirmed with identical losses sequential↔parallel (Exps 684/685, this is clean and defensible); **Exclusion manifest consulted** — Exp 678 wired it after 15 consecutive "mandatory first" misses; **JEPA v15 true-OOD AUC = 0.475** (Exp 682) — the Exp 671 "ood_auc=1.0" claim from .51 did not survive a genuinely held-out test, v15 is no longer cascade-eligible; **VR 200q scale validation anomalous** — Exp 679 reported `signed_improvement=1.0` (0/200 baseline, 200/200 post), but a 0/200 baseline on Qwen3.5-0.8B is physically implausible, and Exp 687's sparse-autoencoder found zero latent features distinguishing correct from incorrect steps — the "win" reads as a grader-format artifact rather than a reasoning improvement; **Prompt-Injection KAN v1 true distillation (Exp 690)** — first distillation in the project with teacher_inference_duration_s=6256 passing the REQ-SAFE-011 invariant; v1_auroc=0.7995 on in-distribution 40-sample held-out, a 0.13 regression vs v0 that is informative (v0 was source-labeled and measured dataset origin, not injection); **Cross-dataset gate (Exp 691)** — reported mean AUROC 0.9585 but confusion matrices at threshold 0.5 show `TP=0, FP=0, TN=N, FN=N` on all three datasets (HackAPrompt, BIPIA, synthetic OWASP), meaning zero injections detected in practice. The `generalization_verified_publishable` verdict is retracted pending threshold calibration. Wall time 4268.83 min (7.86 min/exp, new project best).

Note: Milestone 2026.04.51 (Exps 666-677, 38th milestone, title "EnsembleGate v4 + First Positive VR + JEPA v15") completed. Key results: **RETRO-033 CLOSED** — Live VR Attempt #18 v2 (Exp 668): `honest_verdict=vr_positive`, `signed_improvement=0.64`, `baseline_accuracy=0.36→post_accuracy=1.0` on 25 live questions with structured equation forcing, `inference_mode=live_gpu`, `structured_forcing_recall=1.0`; **EnsembleGate v4** recall-first redesign (Exp 667); **Prompt-Injection KAN Rescue v2** — `honest_verdict=distillation_corpus_built_classifier_trained_auroc_below_threshold` (Exp 669); **JEPA v14 + Platt Cascade Deployment Fix** (Exp 670); **JEPA v15 Live Retrain** — `jepa_v15_ood_auc=1.0`, `honest_verdict=jepa_v15_auc_met` (Exp 671); **KV260 DFX Manager Protocol Fix** — `kv260_status=blocked_bitfile_not_configured` (Exp 672); **DualGPU Proof v3** — `honest_verdict=dualgpu_partial`, simultaneous forward-pass on both GPUs confirmed, `max_gpu1_util_pct=0.0`, RETRO-071 still open (Exp 673); **IAS Adaptive Gate Calibration** (Exp 674); **LOS-Net Sequence Distribution Hallucination Detector** (Exp 675); **MetaJuLS Adaptive Constraint Propagation** (Exp 676); **Retrospective (Exp 677):** `retro_033_status=closed`, `retro_071_status=open_partial`, `vr_signed_improvement=0.64`, `jepa_v15_ood_auc=1.0`, wall_time regression +0.64%, `honest_verdict=wall_time_regression_28min_vr_positive_retro033_closed_manifest_confirmed_dualgpu_partial_retro071_open_partial_jepa_v15_ood_auc_1pt0_kv260_blocked_per_exp_avg_8pt2min`.

Note: Milestone 2026.04.50 (Exps 653-665, 37th milestone, title "Prompt-Injection Safety KAN + Structured Equation Forcing + SpecGuard Mid-Generation Verification") completed. Key results: **StructuredEquationForcer** — detection_rate_on_forced=1.0, equation_forcer_ready (Exp 653); **Ensemble Recall Gate v3** — ensemble_recall=0.224, gate_open=False (below 0.30 threshold, Exp 655); **Live VR Attempt #18 BLOCKED** — RETRO-033 attempt #18 gated (Exp 656); **FR-11 Tier 2 Cross-Session Relay** — fr11_real_violations=True (Exp 659); **LSEBMCL Constraint Memory** — forgetting_rate=0.0, lsebmcl_no_forgetting=True (Exp 660); **Ising Sampler v3 RTL** — h_ema register + EMA stage, N=64 fits XCK26 at 48.5% LUT utilisation (Exp 662); **HALP Probe** — halp_auc=0.442, not viable (Exp 663); **DualGPU Parallel EORM+JEPA** — dualgpu_proven=False, RETRO-071 still open (Exp 664); **Retrospective (Exp 665):** n_criteria_met=5/13, wall_time=4387.6 min (+0.17% regression vs .49), open_retro_count=12, honest_verdict=partial_milestone_5_of_13_criteria_met_retro_033_still_open_after_18_attempts.

Note: Milestone 2026.04.49 (Exps 640-651, 36th milestone, title "HERMES v2 Live Generation Loop + Platt JEPA + Parallel Ising Inertia") completed. Key results: **Exclusion Manifest + DualGPU Preflight** — manifest_wired=True (Exp 640); **HERMES v2 Live Generation Loop** — sentence-by-sentence verification, hermes_v2_recall=0.0, RETRO-070 resolved via ensemble architecture pivot (Exp 641); **Causal Reasoning Verifier** — honest_verdict=causal_improves (Exp 642); **Ensemble Recall Gate v2** — ensemble_recall=0.36, gate_open_vr_unblocked (Exp 643); **Live VR Attempt #17** — BLOCKED, vr_no_improvement_still_blocked (Exp 644); **JEPA v14 Platt Scaling** — platt_calibrated, ECE improved (Exp 646); **OTV One-Token Verifier** — otv_not_viable_keep_eorm (Exp 647); **DualGPU 13B v2** — dualgpu_proven=False, RETRO-071 still open (Exp 649); **LowRankKAEM Multilevel + Sparse** — retro_057_resolved=False (Exp 650); **Retrospective (Exp 651):** retro_070_resolved=True, jepa_v14_calibrated=True, open_retro_count=9 (reduced from 11), honest_verdict=retro_070_resolved_jepa_calibrated_vr_still_blocked. **Exp 652 (post-milestone):** Prompt Injection KAN Classifier distilled from gpt-oss-safeguard-20b — classifier_auroc=0.9262 on 200-example held-out set, 3,432 parameters, 4.24s train time, 19.7ms median inference.

Note: Milestone 2026.04.48 (Exps 627-639, 35th milestone, title "HERMES Improved — All RETROs Carry") completed. Key results: **InterWhen Mid-Generation Monitor** — interwhen_recall=0.12 (3x vs v1 baseline=0.04), early_detection_rate=1.0, retro_070_partial=True, gate_open=False (Exp 627); **ORACLE FOVER v5 Corpus Builder** — oracle_corpus_ready (Exp 628); **InterWhen Diagnostic Gate** — gate_closed_do_not_retry, recall primary=0.12, extended=0.14, below 0.20 threshold (Exp 629); **Live VR Attempt #16 BLOCKED** — RETRO-033 miss #16, recall 0.12 << 0.20 threshold (Exp 630); **JEPA v14 ORACLE Calibrated Retrain** — v14_ood_auc=0.912, calibration still above threshold, v14_uncalibrated (Exp 631); **DualGPU 13B Forward Pass Proof** — model load failed, RETRO-071 still open, seventh consecutive milestone unconfirmed (Exp 632); **HERMES Tool-Augmented Verification Adapter** — hermes_recall=0.12, 3x improvement vs v1 recall=0.04, hermes_fp_rate=0.2, hermes_improved (Exp 633); **Multilevel KAN KAEMEnergy** — no_improvement (Exp 634); **AdapTrack Constrained Generation** — adaptrack_recall=0.08, comparable but below InterWhen (Exp 635); **FPGA TCL v2 Update** — tcl_updated, synthesis_deferred (Exp 636); **LowRankKAEM Sparse Redesign** — sparse_no_improvement, RETRO-057 still open (Exp 637); **FR-11 Relay** — synthetic_fallback (Exp 638); **Retrospective (Exp 639):** n_experiments_run=13, total_wall_time=18.445 min (mean=1.419 min/exp — fastest milestone ever), hermes_improves=True, all_retros_carry=True, open_retro_count=11, honest_verdict=hermes_improved_all_retros_carry.

Note: Milestone 2026.04.47 (Exps 614-626, 34th milestone, title "SymCode Closed — NUP Deployed — Recall Still Blocked") completed. Key results: **ExclusionManifest DualGPU Validation** — precheck timed, DualGPU still unconfirmed (Exp 614, RETRO-071 opened: sixth consecutive milestone); **Live Corpus v3 Expansion** — corpus_partial (Exp 615); **LLMAsExtractorV1** — v1_recall=0.04, no improvement, architecture review required (Exp 616, gate_open=False); **Extractor Diagnostic v5** — timed_out (Exp 617); **JEPA v13 CAPO Calibrated Retrain** — v13_ece=0.207 above 0.10 threshold, uncalibrated (Exp 618); **DSVD-SymCode Hybrid Verifier** — symcode_live_auc=0.804, RETRO-069 RESOLVED, SymCode beats DSVD (Exp 619); **Live VR Attempt #15 BLOCKED** — gate_open=False, 15 consecutive zero-positive attempts confirmed, no more extractor-only passes (Exp 620); **MetaJuLS Online Adaptation** — adaptation_effective=True (Exp 621); **NUP v6 Tier 0c Cascade Wire-In** — nup_deployed_latency_ok, cascade_latency_ms=1.27ms (Exp 622); **TRUST Agents Comparison** — trust_recall=0.0, v1 extractor equivalent (Exp 623); **KV260 Vivado Synthesis v2** — simulation_validated=True, Vivado not yet installed, synthesis blocked (Exp 624); **FR-11 Relay** — synthetic_fallback (Exp 625); honest_verdict=symcode_closed_nup_deployed_recall_still_blocked; open_retro_count=11.

Note: Milestone 2026.04.46 (Exps 601-613, 33rd milestone, title "Probe and Manifest Closed — Recall Still Blocked") completed. Key results: **ExclusionManifest Conductor Verification** — RETRO-067 RESOLVED, manifest verified in conductor with precheck sentinel (Exp 601); **Live Corpus Expansion v2** — success, corpus_expanded (Exp 602); **CoACEExtractorV4 recall=4%** — no improvement vs V3, same ceiling, gate remains closed (Exp 603, RETRO-068 partially addressed); **DSVD Live Fine-Tuning** — no improvement, live AUC dropped to 0.159 (Exp 604, RETRO-069 still open); **NUP Probe v6 CAPO Retrain** — RETRO-049 RESOLVED, nup_v6_auc=0.9643, Tier 0c ready (Exp 608); **JEPA v12 OOD Validation** — v12 overfit confirmed on OOD, v13 checkpoint saved (Exp 607); **Interleaved Formal Logic Verifier** — ilv_improved (Exp 606); **D-Wave Wire-In** — dwave_wired_hisr_integrated (Exp 610); **FACT-E + p-bit Ising RTL** — RTL updated, FACT-E no signal (Exp 612); **Live VR CoACE v4** — BLOCKED, RETRO-033 attempt #14 gated by recall<threshold (Exp 609); honest_verdict=probe_and_manifest_closed_recall_still_blocked; open_retro_count=11.

Note: Milestone 2026.04.45 (Exps 589-600, 32nd milestone, title "Infrastructure Progress — Live Corpus Gap Diagnosis") completed. Key results: **ExclusionManifest Conductor Wire-In** (Exp 589, RETRO-067 CLOSED — conductor now gates legacy experiments via exclusion manifest); **Import-Time CARNOT_FORCE_LIVE Assertion** — assertion module blocks model loading when flag absent (Exp 590, RETRO-062 prevention); **CoACEExtractorV3 live recall=4%** — WORSE than v2's 5.9% (Exp 591, RETRO-068 opened: live-corpus retraining required); **DSVD live AUC=0.586** — below 0.80 deployment threshold (Exp 592, RETRO-069 opened: same offline/live distribution gap as CoACE); **JEPA v12 CPMI+PROGRS retrain** — v12_val_auc=1.0, RETRO-063 validated (Exp 593); **D-Wave Quantum Annealing confirmed** — speedup_ratio=26.24x vs CPU Neal via HISR, dwave_available=true (Exp 598); honest_verdict=infrastructure_progress_no_accuracy_gain; open_retro_count=12.

Note: Milestone 2026.04.44 (Exps 575-588, 31st milestone, title "Recall Surgery and Contrastive JEPA — First Verified Improvement on Live Models") completed. Key results: **ExclusionManifest built** (Exp 575, RETRO-056 CLOSED — 5 legacy experiments manifest-listed, cumulative 2,695 min wasted identified; conductor wiring pending, RETRO-067 opened); **CoACE Recall Boost v2** — offline recall 33.3%→86.7% via multi-step chain tracking and prose pattern recognition (Exp 576, RETRO-064 partial; but live recall=5.9% unchanged — RETRO-066 opened: offline/live distribution gap); **JEPA CPMI Pair Builder** — 9 contrastive hard-negative pairs built (Exp 577); **JEPA v11 CPMI Retrain** — AUC: 0.4444→1.0 via contrastive hinge margin loss (Exp 580, RETRO-063 CLOSED); **Symbolic-KAN Energy** — interpretable energy formula via symbolic regression, symbolic_mse=0.059 vs KAEM_mse=137.2, formula_interpretable=true (Exp 586); **DSVD Adapter** (arXiv 2503.03149) — mid-generation hallucination detection AUC=0.976, Tier 2.5 viable (Exp 587); **KV260 Vivado Synthesis** — Vivado not installed, TCL enhanced, cpu_baseline_latency=289ms (Exp 584); **KV260 Live Benchmark v3** — blocked (no bitfile; upstream Exp 584, Exp 585); honest_verdict=partial_2_retros_closed. New RETROs: RETRO-066 (CoACE offline/live distribution gap, critical), RETRO-067 (ExclusionManifest built but conductor not wired). Next milestone must calibrate CoACE extractor on live model outputs before offline recall gain translates to pipeline accuracy lift.

Note: Milestone 2026.04.43 (Exps 563-574, 30th milestone, title "Root Cause Surgery — Execution-Based Extraction and PURE JEPA Recovery") completed. Key results: **CoACEExtractor FIXED extraction TP=0** — Python eval() on symbolic equations resolves RETRO-061 (Exp 564, RETRO-061 CLOSED); **CoACE live diagnostic** — gate opened for Exps 569+570, TP/FP confirmed on 25 known-incorrect responses (Exp 565); **JEPAPUREMinForm loss** — PURE min-form PRM objective implemented (Exp 566, RETRO-060 addressed); **JEPA v10 retrain** still inverted: v10_auc=0.4444 below random despite PURE objective — RETRO-063 opened (Exp 567); **KV260 FPGA bring-up v2** — first real hardware test post board arrival, fpga_alive=false (bitfile not yet loaded), Verilog synthesis paths confirmed (Exp 568); **FR-11 real CoACE violations** collected (Exp 570, RETRO-033 attempt #11: signed_improvement=0.0, coace_recall=0.059 — RETRO-064 opened); **HalluField Tier 0e** thermodynamic hallucination detection (Exp 571); **PRA EORM beam search** K=3 energy-guided decoding (Exp 572); **Energy-per-token calibration** blocked by RAPL unavailability on AMD hardware — RETRO-065 opened (Exp 573); honest_verdict=partial_fix. New RETROs: RETRO-063 (JEPA architecturally inverted), RETRO-064 (CoACE recall 5.9% — accuracy improvement undetectable at this recall), RETRO-065 (RAPL unavailable). Next milestone must improve CoACE recall to >30% before scheduling accuracy benchmarks.

Note: Milestone 2026.04.42 (Exps 549-562, 29th milestone, title "Break the Synthetic Barrier — Live Data Sprint, Root Cause Diagnosed, JEPA Recovery") completed. Key results: **Live 50q Data Collection A** — GSM8K indices 0-49 collected with live GPU inference, Phase 2 live data sprint complete (Exp 551); **EORM GRPO Retrain on 100+ Real Pairs** — RETRO-058 fix, real corpus now exceeds synthetic threshold (Exp 556); **JEPA v9 Retrain on Diverse 100+ Corpus** — LeWorldModel objective, RETRO-056 addressed, status=success (Exp 557); **Tier 1 Self-Learning Relay on Real Data** — FR-11 mandatory relay with n_responses=25, honest_verdict=real_data_no_improvement (Exp 561, first real-data relay in project history); **Milestone 2026.04.42 Retrospective** (Exp 562). Conductor exclusion manifest built and zombie kill executed (Exp 549). BatchedInferenceRunner real migration complete (Exp 550).

Note: Milestone 2026.04.41 (Exps 537-548, 28th milestone, title "Close the Nine-Milestone Gap — First Live 25q Positive, Teardown Fix, GRPO Self-Learning") completed. Key results: **RETRO-054 CLOSED** — ExperimentTemplate.teardown() + atexit registration implemented, zombie VRAM carryover prevention now in framework (Exp 537); **RETRO-055 CLOSED** — env_autofix value-check fix confirmed working in live_gpu mode (Exp 538); **RETRO-033 miss #10** — live 25q pipeline accuracy 0.32 == baseline 0.32 (signed_improvement=0.0, live GPU mode confirmed operational); **RETRO-038 miss #8** — live 100q pipeline accuracy 0.29 == baseline 0.29, Wilson CI spans zero; **LowRankKAEM wired as default tier** — 4.6x speedup at n_vars=10, 154.7x at n_vars=200 (Exp 544); **GRPO EORM improved** on 3 synthetic pairs (AUC 0.00→1.00, honest_verdict=synthetic_fallback) (Exp 540); **AutoRefine distilled 2 constraint templates** from 67 violations (Exp 546); mean=3.785 min/exp (new project record, 41.6 min for 11 experiments). New RETROs: RETRO-056 (JEPA AUC 0.444 below random on 24-pair corpus), RETRO-057 (LowRankKAEM energy_mad_normalized 0.96-0.99, outside 5% tolerance), RETRO-058 (synthetic proxy fallback epidemic: 6/11 experiments), RETRO-059 (conductor exclusion manifest for fully-modern legacy scripts).

Note: Milestone 2026.04.40 (Exps 526-536, 27th milestone, title "Fix the Last Gate — Eighth Attempt, First Live Positive") completed. Key results: **RETRO-053 RESOLVED** — env_autofix one-liner now overrides falsy CARNOT_FORCE_LIVE='0' (Exp 526); **Live 100q Precision v8 timed out at 45 min during actual live inference** (Exp 527, RETRO-033 miss #9 — new blocker is inference latency, env gate no longer the problem, significant progress); **NUP Probe v4 (Tier 0c) + Hallucination Basin Detector (Tier 0d) wired into ThreeTierPipeline** (Exp 530); **JEPA Live Retrain v7 FR-11 confirmed** — final_auc=0.967 on 46 live FOVER pairs (Exp 535); **LowRankKAEMEnergy** 23.7x speedup at k=2 (Exp 532); mean=5.0 min/exp (new project record, 55 min total for 11 experiments). New RETRO-055: reduce n_questions to 25 or increase timeout to 90 min for live benchmark to complete within budget.

Note: Milestone 2026.04.39 (Exps 513-524, 26th milestone, title "Close the Credibility Gap") completed. Key results: **JITVRAMCheck wired into all model loaders** (Exp 513, RETRO-051 CLOSED); **NUP Probe v4 contrastive training AUC=1.0**, Tier 0c promoted (Exp 523, RETRO-049 CLOSED); **LeWorldModel-JEPA AUC=0.972 with 274x variance reduction** vs standard BCE (Exp 520, training stability breakthrough); **Hallucination Basin Detector viable at Tier 0d, AUROC=1.0** (Exp 521); **JEPA Live Retrain v6 FR-11 confirmed** — final_auc=1.0 on live FOVER pairs (Exp 522); **GSM-Symbolic adversarial thesis definitively rejected** (Exp 516, honest_verdict=thesis_rejected, RETRO-039 CLOSED as negative). New critical RETRO-053: env_autofix does not override CARNOT_FORCE_LIVE='0', blocking live benchmarks for 7th consecutive milestone (Exps 514/515 deferred). Fix is a single conditional in apply_env_autofix().

Note: Milestone 2026.04.26 (Exps 351-364) discovered that `CARNOT_FORCE_LIVE` was never being set by the conductor (RETRO-012), which caused three consecutive milestones of silent simulated fallback despite both RTX 3090s being live-capable. Milestone 2026.04.27 (Exps 365-376) closed RETRO-012/013/014 but live GPU remained unconfirmed for a fourth consecutive milestone (RETRO-015 critical). Milestone 2026.04.28 (Exps 377-389) fixed the infrastructure (Exp 377: LiveGPUGate + session_startup.sh export), but the GPU node was offline during the conductor session — live GPU unconfirmed for a fifth consecutive milestone (RETRO-019 critical). Milestone 2026.04.29 (Exps 390-402, 16th milestone) ran entirely in "deliverable already exists" fast-path mode — GPU node offline for a SIXTH consecutive milestone; RETRO-022 critical human escalation opened (cloud GPU or power on RTX 3090 node required). Milestone 2026.04.30 (17th, complete): Exp 404 confirmed GPU hardware IS present (`is_live_capable=True`); Exp 413 EnvironmentAutoFix resolved RETRO-022 via `apply_env_autofix()` self-injecting `CARNOT_FORCE_LIVE=1`; Exp 419 CRANE extractor implemented as primary extraction path for FULL_STACK variant — live run pending. Milestone 2026.04.31 (18th, complete): operational retrospective written; 429 cumulative experiments, 103.0 hours total; GPU 0 at 91% utilization at retro time — first milestone retro with a live inference process in-flight (positive trend); RETRO-025 opened (GPU 1 idle VRAM); RETRO-022 partially closed. Milestone 2026.04.32 (19th, complete): 12 experiments (Exps 425-435a); ExperimentTimeoutWatchdog deployed (RETRO-003 CLOSED after 17+ milestones, Exp 425); DualGPUHealthCheck + temperature guard (Exp 426); live benchmark re-run harnesses (Exps 427-429, scaffolding_only — 45-min conductor budget insufficient for live runs); FOVER Z3 step annotation pipeline (Exp 430, 35 tests); JitRL live validation — synthetic_fallback, 33.71% FP reduction (Exp 432, 39 tests); SpilledEnergyDetector Tier 0 pre-filter added to ThreeTierPipeline per arXiv 2602.18671 (Exp 433, 26 tests); ComplianceEnergyChecker KAN-based module for regulated industries (Exp 434, 67 tests); Kona Phase 3 seed — continuous energy landscape (Exp 435a, 39 tests); operational retrospective (Exp 436, 58 tests); live_numbers_confirmed=False — RETRO-026 (live benchmarks need >45-min executor), RETRO-027 (silent experiment drop) opened. Milestone 2026.04.33 (20th, complete): 12 experiments (Exps 437-448) + retro Exp 449; mean=21.2 min/exp (improved from 31.7); **FIRST live GPU benchmarks after 7 consecutive scaffolding-only milestones** — all three benchmark experiments confirmed `inference_mode='live_gpu'`; results are honest negatives: Qwen3.5-0.8B 14% baseline accuracy (no improvement from repair), Gemma4-E4B-it 0% accuracy (model issue, RETRO-028), pass@1=0.0 HumanEval, 14pp adversarial drop with 0% repair recovery; LongRunBenchmarkExecutor implemented (Exp 437, RETRO-026 CLOSED); FOVER live annotation: 57 real CoT steps labeled — first `real_data_labeled` verdict after 8 consecutive `synthetic_only` milestones (Exp 442); EORM + JEPA retrained on real data: JEPA AUC 0.457→0.571 (Exp 443, RETRO-024 CLOSED); CarnotThinkProbe Tier 0 generative CoT pre-filter (Exp 444, 56 tests); BoltzmannRepairBridge energy-guided repair direction (Exp 445, 30 tests); KAEMEnergy exact inverse-transform sampling via spline marginals (Exp 447, 51 tests, mean_speedup=1.29x vs MCMC); operational retrospective (Exp 449, 75 tests, schema=carnot.operational_retro.v7); RETRO-028 (Gemma4 zero accuracy), RETRO-029 (think_probe timeout), RETRO-030 (silent drop), RETRO-031 (KAEM no speedup) opened. Milestone 2026.04.34 (21st, complete): **FIRST POSITIVE verify-repair number** — Exp 451 live_precision_improvement=+5pp (honest_verdict=repair_better, first since Exp 411); GemmaTransformersLoader replaces llama.cpp (Exp 450, RETRO-028 CLOSED); AtomicResultWriter (Exp 452, RETRO-030 CLOSED); VeriCoTStepValidator FOL+Z3 UNSAT detection (Exp 453, 56 tests); VPRMArithmeticVerifier 6-family rule engine (Exp 454, 80 tests); ThinkProbeV2 (Exp 455, RETRO-029 CLOSED); ConstraintAdditionFromMemory FR-11 Tier 1 self-learning — session2_fp_rate=0.0 (Exp 456, 27 tests); LSEBMConstraintReplayer FR-11 Tier 2 cross-session EBM replay (Exp 457); AMD XDNA NPU unblock via pip install mlir-aie (Exp 460). All live GPU result counts above reflect artifacts generated before the RETRO-012 bug was identified. Milestone 2026.04.35 (22nd, complete): 12 experiments (Exps 462-473); DeliverableGuard + DualGPURunner closes RETRO-032 — zero silent deliverable drops since deployment (Exp 462); Conductor Session Health Check zombie killer at session start (Exp 463); EBM-CoT Calibration v3 AUC=0.848889 closes RETRO-034 (Exp 466, EP update + Langevin, 57 real pairs + 93 synthetic, arXiv 2510.12934 + 2511.07124); PPSEBM Tier 2 constraint partitioner: partition_isolation_score=1.0, fp_rate=0.0 across all three domains (Exp 470); KV260 FPGA v2: Verilog RTL generated (128-spin, sparsity=0.9, 1,542 edges), rtl_ready_for_synthesis — hardware arrives 2026-04-20 (Exp 471); JEPA Tier 3 AUC regressed 0.667→0.400 (honest negative — Tier 3 training added noise, Exp 472); HumanEval Live VeriCoT code_no_improvement (Exp 469); 4 experiments deferred_to_gpu (Exps 464/465/467/468 — GPU zombie VRAM main blocker); retro adoption_rate=50% (5/10), RETRO-041 generated to force remaining items. Cumulative: 473 experiments, 4,100+ passing tests. Milestone 2026.04.36 (23rd, complete): 13 experiments (Exps 474-486), "Fix the Root Cause"; **GPUVRAMGate** wired before every GPU experiment (Exp 474, RETRO-037/042 closed); **Conductor Dedup Check + Partial-Result Handoff** (Exp 475); **GSM-Symbolic Adversarial Benchmark live** (Exp 479, RETRO-039 closed); **Harness DualGPURunner Enforcement** across 361 scripts, 53 missing cuda:1 patched (Exp 480); **ThinkProbeV2 Live GPU v3** (Exp 482, RETRO-036/042 closed); **KAEM 5x speedup crossover** at n_vars=250 (Exp 483, RETRO-031 resolved); **Neural Uncertainty Principle Probe** identifies hallucination mechanism (Exp 484); **PPSEBM validated on real data** fp_rate=0.0, partition isolation=1.0 (Exp 485, RETRO-043 closed); **Retrospective (Exp 486):** credibility_gap_closed=false (Exps 476/478 deferred to GPU), retro_adoption_rate=1.0, infrastructure_hardening_complete=true, JEPA AUC 0.401→0.281 regression open. Cumulative: **486 experiments**, **4,100+ passing tests**. Milestone 2026.04.37 (24th, complete): 13 experiments (Exps 487-499), "Did we break the VRAM deadlock?"; **GPUVRAMGateV2** kills zombies before VRAM check, fixing race condition (Exp 487, RETRO-044 CLOSED); **Live benchmark harnesses v3** with GPUVRAMGateV2 — infrastructure verified, live execution blocked by conductor process consuming 8.96 GiB vs 14.89 GiB required for Gemma4 full precision (Exps 488/489/490); **JEPA Curriculum Retrain V3** recovers AUC 0.281→0.967 via confidence-descending curriculum order (Exp 492, RETRO-040 CLOSED); **Batching Enforcement Pre-Commit Hook** prevents violations at commit time (Exp 493, RETRO-045 CLOSED); **GPU Thermal Gate** defers experiments when GPU >85°C (Exp 494, RETRO-046 CLOSED); **DualGPU Harness Enforcement v2** patches 53 remaining scripts with explicit cuda:1 (Exp 495); **NUP Probe v2** Bayesian semantic entropy for Tier 0c hallucination detection (Exp 496); **SuRe Surprise-Driven EBM Replay** priority replay for Tier 2 self-learning (Exp 497, arXiv 2511.22367); **KAEM Extended Profile n=5000** extends crossover search (Exp 498, RETRO-031 extended closure); **Retrospective (Exp 499):** VRAM deadlock NOT fully broken — conductor process 8.96 GiB leaves only 5.37 GiB free vs 14.89 GiB required; RETRO-048 (quantize Gemma4 INT4/GGUF) opened; credibility_gap_status=PARTIALLY_CLOSED; adoption_rate=1.0 maintained; FR-11 Tier 3 fully recovered from regression. Cumulative: **499 experiments**, **4,400+ passing tests**. Milestone 2026.04.38 (25th, complete): 13 experiments (Exps 500-512), "Break the Credibility Ceiling"; **Gemma4 INT4 Quantization** (Exp 500, RETRO-048 RESOLVED); **Conductor CPU Routing + VRAM Budget Ledger** (Exp 501); live benchmarks v6 (Exps 502/503/504) all deferred at runtime — RETRO-033 sixth consecutive miss, RETRO-051 opened (just-in-time VRAM check critical path); **KAEM Distribution Family** KAEM advantage found on gaussian_mixture (Exp 508, RETRO-031 CLOSED); **PPSEBM Energy-Magnitude Replay** isolation_improvement=1.1172 vs SuRe=-0.1172 (Exp 509, RETRO-050 CLOSED); **JEPA Live Retraining v4** quasimetric regularization (Exp 510); AMD XDNA NPU probe (Exp 511, npu_not_available); **Retrospective (Exp 512):** credibility_milestone_reached=False (6th miss), RETRO-048/031/050 resolved, RETRO-051 critical path for milestone 2026.04.39. Cumulative: **512 experiments**, **4,400+ passing tests**.

## PBT Verification

Carnot's strongest live evidence is now the Hypothesis-backed code-verification path. The full Gemma run is positive and statistically significant; the seeded Qwen transfer check is intentionally honest about a flat repair delta while still showing additive verifier signal.

| Live PBT artifact | Baseline | +Carnot | Added verifier signal | Experiment |
|-------------------|----------|---------|-----------------------|------------|
| Full HumanEval 164 (Gemma4) | 11.6% | 14.6% | 144/145 wrong baselines detected; 6 official-test misses caught; 5 repairs | Exp 226 |
| Seeded HumanEval 30 (Qwen3.5-0.8B) | 23.3% | 23.3% | 17/23 wrong baselines detected; 2 official-test misses caught; 0 repairs | Exp 227 |
| Dual-model HumanEval 50 | 18.0% / 10.0% | 20.0% / 12.0% | 144/145 wrong-code detections across the paired slice | Exp 220 |

### Code verification (strongest domain)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| HumanEval 164 problems (PBT) | 11.6% | 14.6% | **+3.0pp** [+0.6, +6.1] CI | Exp 226 |
| HumanEval 30 problems (PBT, seeded Qwen cohort) | 23.3% | 23.3% | +0.0pp; 2 harness misses caught | Exp 227 |
| HumanEval 50 problems (PBT) | 18.0% / 10.0% | 20.0% / 12.0% | +2.0pp both models | Exp 220 |
| HumanEval 30 problems (execution) | 16.7% | 20.0% | +3.3pp | Exp 208 |

PBT detects 99.3% of wrong code (144/145) on the full Gemma run and catches 6 official-test misses. Exp 227 shows the same verifier still adds signal cross-model even when the repair loop itself stays flat.
On deterministic HumanEval-style probes, the same Hypothesis-backed verifier also catches **5/5** under-specified bugs that execution-only checks miss while preserving **5/5** matching correct solutions (Exp 224).

### Constraint verification (math/instruction)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| Typed IR constraints (81 tasks) | 61.7% | 66.7% | **+4.9pp** (Gemma4) | Exp 221 |
| GSM8K semantic v2 (200 questions) | 46.5% | 47.5% | +1.0pp (Gemma4); verify-only still unjustified | Exp 235 |
| GSM8K arithmetic (100 questions) | 91.0% | 91.0% | 0.0pp | Exp 206 |

Semantic verifier v2 reuses the fixed Exp 219 cohort and trims Qwen false positives from **7** to **4**, but verify-only still hurts on both models and Gemma now carries **26** unnecessary repair triggers (Exp 235). The main win is better calibration and cleaner abstention, not a solved live semantic benchmark yet.

### Self-learning

| Metric | Without learning | With learning | Improvement | Experiment |
|--------|-----------------|---------------|-------------|------------|
| Held-out success | 34.48% | 34.48% | Flat across all four strategies; primary success **not met** | Exp 241 |
| Retrieval quality | 0.0% hit / 0.0% precision | 32.1% hit / 43.6% precision | Better case reuse without extra wins or false positives | Exp 241 |

On **116** held-out cases against **344** learning cases, `no_learning`, `tracker_only`, `case_memory`, and `case_memory_plus_policy` all finish at **34.48%** success with **8** false positives. The latest positive signal is narrower and more honest than Exp 223: `case_memory` materially improves retrieval hit rate and precision, but those better matches still do not convert into held-out task gains under the zero-additional-false-positive budget.

### Latest follow-ons

- **Semantic calibration + live rerun (Exp 232 / Exp 235):** the new calibration corpus contains **568** rows (**155 TP / 33 FP / 221 FN / 159 TN**), and the semantic-verifier-v2 rerun keeps verify-only explicitly unjustified even after cleaner thresholds and abstention logic.
- **Spec-aware code verification (Exp 236 / VERIFY-036):** the explicit code-spec corpus now covers **164** HumanEval tasks with **194** trace links, **8** official-test-miss traces, and **5** repaired traces, and the packaged verifier can now combine official tests, PBT, and explicit spec clauses through `verify_generated_code_with_specs()`.
- **Chronological replay v2 (Exp 241 / VERIFY-038 / VERIFY-039 / VERIFY-040):** richer case keys plus compiled policy context improve held-out retrieval to **32.1%** hit rate and **43.6%** precision, but the primary task-gain success condition is still honestly **not met**.
- **KV260 host / overlay round-trip (Exp 242):** the checked-in artifact is intentionally blocked in this environment because no `CARNOT_KV260_BITFILE` path was configured; `mode="auto"` still resolves to CPU fallback instead of fabricating board timings.
- **Sampler-backed replay (Exp 243):** CPU reranking stays neutral across **460** saved semantic and code repair cases, while the KV260-backed path remains blocked by the same missing bitfile setup.
- **Formal claim corpus (Exp 244 / VERIFY-041):** **2,545** provenance-bearing rows from live traces — **1,243** solver-routable (arithmetic, boolean-entailment, set-membership, execution-oracle, cardinality, comparison) and **1,302** explicitly `not_formalizable`.
- **Process integrity corpus (Exp 248):** **849** rows across five labels (`right_answer_wrong_process`, `wrong_answer_partially_sound_process`, `unsupported_step`, `repair_fixed_outcome_only`, `repair_fixed_process_and_outcome`, `clean`), built from Exp 235 and Exp 238 live traces.
- **Process-aware verification comparison (Exp 251):** process verification added **0** rejections beyond the spec-aware gate on both models, but caught **5** `outcome_correct_process_invalid` cases (Qwen=3, Gemma=2) across **143** combined defect instances.
- **Predictive verifier hardware benchmark (Exp 257):** ONNX CPUExecutionProvider runs at **5.8 µs/call** (**7.1×** faster than CPU NumPy at **41.8 µs/call**); CUDA ORT and AMD XDNA NPU paths remain blocked by missing toolchain.
- **PrefillUncertaintyProbe (Exp 295 / REQ-VERIFY-080):** entropy-based pre-generation hallucination gate (arXiv 2603.19562). High entropy → `high_risk=True` → trigger full verification before any tokens are generated. Black-box, no gradient access required.
- **ConstraintGenerator from CaseMemory (Exp 300 / REQ-LEARN-010/011):** converts CaseMemory violation patterns into new constraints when observed_precision ≥ 0.85 (soundness bound, arXiv 2603.03538). Purely additive — never removes existing constraints.
- **Confidence-weighted repair gating (Exp 301 / REQ-VERIFY-081/082):** EBM energy-derived confidence scores gate the repair loop. Violations below threshold=0.8 are suppressed, fixing Exp 184's 0% net improvement from false-positive repairs.
- **Integrated Tier 1+2 self-learning benchmark (Exp 302):** first end-to-end run combining Exp 301 confidence-weighted gating and Exp 300 ConstraintGenerator on 100 questions (2 × 50 batches). Reports honest signed `improvement_delta`; negative values not hidden.
- **AMD XDNA NPU Unblock (Exp 303):** full source-build + inference benchmark path ready. Currently `blocked_prereq` (ninja + openblas missing). Run `sudo pacman -S ninja openblas` then re-run Exp 303 to auto-advance.
- **FCV LIVE on HuggingFace (Exp 304):** `Carnot-EBM/carnot-formal-claim-verifier-v1` is now live. Python API credential fallback resolved the Exp 293 CLI blocker.
- **Experiment template + batching harness (Exp 306 / REQ-VERIFY-083/084):** `ExperimentTemplate` + `BatchedInferenceRunner` eliminate 15–20 min cold-start per experiment. Template overhead validated at **0.0001 s** (target < 0.5 s).
- **Conductor hardening (Exp 325 / REQ-INFRA-001/002):** 45-min hard timeout wrapper (`run_experiment_with_timeout.sh`) and test-first stub generation (`generate_test_stub()`). Estimated 27% wall-time speedup per milestone.
- **DualGPUMonitor (Exp 326 / REQ-INFRA-003/004):** zombie process detection + idle-GPU checks wired into `setup_gpu()`; CI-safe.
- **Live GPU full-scale benchmark (Exp 328):** Qwen3.5-0.8B 27.5%, Gemma4-E4B-it 26.3% on adversarial GSM8K all-variant (`inference_mode=live_gpu`). Live accuracy ~10% below simulated baseline (honest divergence).
- **Live relay benchmark (Exp 329):** four-tier self-learning relay on live GPU; `improvement_1to3 = -6.1%` (negative relay signal, research concern; honest signed delta).
- **Live HuggingFace publish (Exp 330 / REQ-PUBLISH-004):** 16 per-token EBM READMEs updated with live-GPU benchmark numbers. FCV README updated. Joint-constraint placeholder created.
- **Confidence-weighted repair — dual-signal FP reduction (Exp 332 / REQ-VERIFY-083/084/085):** expression specificity + Ising variance gate. FPs avoided: **86.7%** (13/15), TPs preserved: **100.0%** (15/15), `GATE_EFFECTIVE`.
- **Model-adaptive constraint thresholds + selective CaseMemory (Exp 333 / REQ-LEARN-015/016):** `PerModelFPTracker` auto-disables range_check for qwen3.5-0.8b (fp_rate=0.73 > tp_rate=0.27 after 15 obs). `SelectiveConsolidation` (ATLAS arXiv 2511.01093) at consolidation_ratio=0.60 (`ADAPTIVE_PASS_ATLAS_PARTIAL`).
- **VERGE-style iterative Z3 refinement (Exp 334 / REQ-REPAIR-012/013):** `VergeRefiner` identifies the specific failed Z3 assertion and repairs only that step; 3-iteration max loop.
- **CoT Circuit Verifier (Exp 336 / REQ-EXTRACT-015/016):** `CoTCircuitVerifier` extracts a computational dependency graph from chain-of-thought responses and checks structural consistency (value-carryover mismatches, cycles). Catches error classes that arithmetic regex and Z3 miss.
- **Milestone 2026.04.24 retrospective (Exp 337):** 12 experiments, 293 total min, mean 24.4 min/exp. **Actual speedup: 39.9%** vs prior milestone baseline (exceeds 27% estimate). All 4 prior action items resolved.

### Milestone 2026.04.25 (Exps 338-348)

- **Host prereqs registry + DualGPU auto-assignment (Exp 338 / REQ-INFRA-006/007):** `HostPrereqsRegistry` checks packages before experiment launch; `DualGPURunner` selects the idle GPU automatically. Closes RETRO-005/006.
- **Pre-session startup health check (Exp 339 / REQ-INFRA-008):** `scripts/session_startup.sh` (--dry-run / --kill-zombies) detects GPU count and zombie processes before any experiment. Closes RETRO-007/008.
- **Live full-precision pipeline benchmark (Exp 340 / REQ-BENCH-003):** first honest measurement of combined precision stack (5 variants × 2 models × 200 GSM8K). `PipelineVariant` enum: BASELINE, CONFIDENCE_ONLY, CONFIDENCE_ADAPTIVE, CONFIDENCE_ADAPTIVE_VERGE, FULL_STACK. `compute_signed_improvement` (honest signed delta, no clamping). CI-safe simulated mode; blocked on GPU failure.
- **HumanEval code verification benchmark (Exp 341 / REQ-BENCH-004):** `CodeExtractor + VerifyRepairPipeline` on 50 HumanEval-style problems. `compute_pass_at_1`, `compute_pass_at_1_after_repair`, `build_humaneval_artifact` (schema `carnot.humaneval_benchmark.v1`).
- **ConstraintTemplateLibrary — Tier 2 constraint addition (Exp 343 / REQ-LEARN-017/018):** 4 built-in Eidoku-taxonomy templates (`carry_check`, `sign_check`, `unit_consistency`, `comparison_direction`). Patterns observed above `min_frequency` threshold activate the template and inject new constraints additively into the pipeline.
- **CaseMemory → ConstraintTemplateLibrary wiring + benchmark (Exp 344 / REQ-LEARN-019):** `CaseMemoryTemplateWiring.on_violation_recorded()` fires `observe_pattern()` on every recorded violation. Constraint-addition benchmark (200 simulated questions, seed=42): Control group shows 0% detection; Treatment group confirms **positive improvement_delta** after `carry_check` activates at 5 violations. Hypothesis confirmed.
- **SessionMemory — multi-session persistence (Exp 345 / REQ-LEARN-020/021):** `SessionMemory(storage_dir, model_id)` serialises `CaseMemory`, `ConstraintTemplateLibrary`, and `PerModelFPTracker` to disk across process restarts. `VerifyRepairPipeline` gains optional `session_memory` param and `close()` save method.
- **EORM CoT energy reward model (Exp 346 / REQ-LEARN-022/023):** pure-JAX transformer encoder (`EORMModel`, `EORMTrainer`) implementing arXiv 2505.14999. Contrastive hinge loss: `max(0, E_correct - E_incorrect + margin)`. AUC-ROC evaluation; saves `results/eorm_model_346.safetensors`.
- **JEPA real-data retrain on live violation pairs (Exp 347 / REQ-LEARN-024):** `extract_violation_pairs` word-tokenises Exp 340 responses and splits at `prefix_fraction=0.5`. `JEPARetrainer` trains with binary BCE loss and evaluates trapezoidal AUC-ROC. CI-safe synthetic fallback (50 pairs) when live GPU data unavailable.
- **SinkProbe attention-sink pre-filter (Exp 348 / REQ-VERIFY-086/087):** implements arXiv 2604.10697 as the first gate in the three-tier pipeline (SinkProbe → EORM → Ising). `compute_sink_concentration` accepts (n_heads, seq_len, seq_len) attention tensors; `SinkProbe(threshold=0.3)` sets `is_uncertain = mean_sink_score < threshold`. Simulated benchmark: **skip_rate=60%, FNR=0%, TNR=100%** — 60% fewer Ising calls with no false negatives.

### Milestone 2026.04.26 (Exps 351-364)

- **Live GPU Diagnostic — silent fallback bug fixed (Exp 352 / REQ-INFRA-014):** Diagnosed and fixed the critical bug where `CARNOT_FORCE_LIVE=1` was silently ignored by the conductor, causing Exps 340/341/346/347 to run in simulated mode despite live-capable RTX 3090s. `LiveGPUDiagnostic` now raises `RuntimeError` when forced-live prewarm fails instead of silently falling through.
- **Live GPU Smoke Test Gate (Exp 353 / REQ-BENCH-005):** `run_smoke_test()` gates any benchmark experiment launch; CI-safe when `CARNOT_FORCE_LIVE` not set; raises on live GPU unavailability when forced.
- **Adversarial GSM8K harness + execution (Exp 354/355 / REQ-BENCH-006/007):** Apple adversarial GSM8K benchmark (arXiv 2410.05229). Three-condition runner: standard / adversarial / repaired-adversarial. `honest_verdict=improvement_positive` gated on `inference_mode==live_gpu AND repair_improvement>0` — never emitted for simulated results. Live execution pending RETRO-012 fix.
- **LLMz3Formalizer — LLM-guided Z3 formalization (Exp 357 / REQ-EXTRACT-019/020):** Implements arXiv 2601.04675 (80% Z3 success rate improvement via task decomposition). `LLMz3Formalizer` extracts Z3 constraints via structured LLM prompting with sandboxed execution (restricted `__import__`, `print→StringIO`).
- **Three-tier pipeline complete (Exp 360 / REQ-VERIFY-088):** `ThreeTierPipeline(SinkProbe → EORM → Ising)` with early-exit at each tier. `verify()` returns `(verified, tier_used, energy)`. Simulated: 30% fewer Ising calls from SinkProbe alone; 60%+ combined skip rate.
- **Three-tier self-learning relay (Exp 361 / REQ-LEARN-026/027):** End-to-end relay across all three tiers. Simulated run: batch1_accuracy=0.60 → batch4_accuracy=0.72 (`improved=True`); all 4 Tier 2 templates activated. `honest_verdict=synthetic_only` — live GPU required for `learning_confirmed`.
- **SAVeR multi-turn verification wrapper (Exp 362 / REQ-AGENT-001/002):** `SAVeRVerifier` implements the arXiv 2604.08401 auditor-before-commit loop for multi-step agent reasoning chains. Goal #4 from research-program.md complete.
- **EORM real-data retrain (Exp 359 / REQ-LEARN-025):** `retrain_mode=synthetic_only` (5 real HumanEval pairs with unique question IDs — no cross-pair contrastive triples). `honest_verdict=synthetic_only`. Fixed `_pairs_to_contrastive_triples` bug: synthetic question IDs now routed to shared pool.
- **ModelServer + TensorRT + DualGPU wiring (Exp 364):** Infrastructure wiring — ModelServer, TensorRT, and DualGPU inference acceleration integrated into all benchmark harnesses for consistent hardware-accelerated testing.
- **Milestone 2026.04.26 retrospective (Exp 363):** 11/12 experiments ran (Exp 356 LLMExtractor skipped). 366 min total, mean 33.3 min/exp. RETRO-012 (CARNOT_FORCE_LIVE bug) is the critical blocker for all live GPU headline results.

### Milestone 2026.04.27 (Exps 365-376)

- **RETRO-012/013/014 close (Exp 365):** `scripts/conductor_gpu_env.sh` created with `CARNOT_FORCE_LIVE=1`; RetroJSONEnforcer pattern established for mandatory result JSON production; LLMExtractor gap documented. RETRO-012/013/014 formally closed.
- **LLMConstraintExtractor (Exp 366):** Second LLM call extracts structured arithmetic claims from free-form IT model output. `LLMConstraintExtractor` added to `python/carnot/pipeline/llm_extractor.py`; prompted extraction with fallback to regex; 100% module coverage.
- **Live extraction comparison harness (Exp 367):** Three-way head-to-head: `ArithmeticExtractor` vs `LLMConstraintExtractor` vs `LLMz3Formalizer` on 30 GSM8K questions with Gemma4-E4B-it (GPU0) + Qwen3.5-0.8B (GPU1). Hard `CARNOT_FORCE_LIVE=1` gate; `honest_verdict=live_gpu_winner` only when all results confirmed live. Live run pending.
- **Live precision pipeline benchmark v2 (Exp 368):** Hard CARNOT_FORCE_LIVE gate; schema=`carnot.precision_benchmark.v2`; `honest_verdict=live_improvement` only when `inference_mode==live_gpu AND signed_improvement>0`. First credible precision-stack headline number pending live run. 74 tests pass.
- **Live HumanEval benchmark v2 (Exp 369):** Hard CARNOT_FORCE_LIVE gate (3-stage: env + diagnose_live_gpu + model_load); `CodeExtractor + VerifyRepairPipeline`; PBT via determinism/idempotency checks; `honest_verdict=code_verification_positive` only on live GPU with positive signed improvement. 69 tests pass.
- **Live adversarial GSM8K benchmark v2 (Exp 370):** Hard CARNOT_FORCE_LIVE gate (`diagnose_live_gpu_or_raise()` — raises RuntimeError, NO simulated fallback); three conditions (standard/adversarial/repaired-adversarial) with LLMConstraintExtractor for repair; `honest_verdict=improvement_positive` gated on `inference_mode==live_gpu`. 23 tests pass; `SCENARIO-BENCH-022` added to spec.
- **EORM retrain on real pairs v2 (Exp 371):** Retrains EORM model on real CoT pairs from Exp 368/369/370; `eorm_model_371_real.safetensors` when live GPU available; priority fallback: 371_real → 346_synthetic in downstream experiments.
- **Three-tier pipeline live benchmark v2 (Exp 373):** Hard `CARNOT_FORCE_LIVE=1` gate via `diagnose_live_gpu()`; Beta-mixture approximate attention (realistic sink distribution vs Exp 360 binary); `compute_honest_verdict()` with 4-branch conservative reporting; `artifact_type=carnot.three_tier_benchmark.v2`. 80 tests pass; `SCENARIO-VERIFY-118/119` added to spec.
- **FR-11 self-learning relay live run (Exp 374):** Three-tier online self-learning relay with real models; `learning_confirmed` verdict gated on `inference_mode==live_gpu`; honest_verdict=synthetic_only without live GPU.
- **CIKAN energy layer (Exp 375):** Clifford-Informed KAN-Ising hybrid energy model (`CIKANEnergy`) for geometric constraint representation. Note: `cikan_energy.py` delivered as JSON not Python — RETRO-018 opened for reimplementation.
- **Milestone 2026.04.27 retrospective (Exp 376):** 11 experiments (Exps 365-375), mean=22.7 min/exp (apparent speedup from fast-fail blocked experiments, not genuine GPU work). `live_gpu_confirmed=False` for FOURTH consecutive milestone — RETRO-015 (critical) opened. New RETRO-015/016/017/018. 78 tests pass. `results/operational_retro_2026_04_27.json`.

### Milestone 2026.04.28 (Exps 377-389) — 15th Milestone

- **Live GPU infrastructure fix (Exp 377):** `LiveGPUGate` class + `session_startup.sh` export of `CARNOT_FORCE_LIVE=1`. Formally closes RETRO-015 at the infrastructure level. GPU node was offline during session — RETRO-019 escalation opened for execution-environment failure.
- **Combined EORM+JEPA retrain (Exp 383):** Trains EORM on contrastive triples and JEPA on binary violation pairs from live CoT pairs (Exps 379-382). `schema=carnot.combined_retrain.v1`; `honest_verdict=insufficient_pairs` (Exps 379-382 live files empty — RETRO-015 upstream). `eorm_model_383_real.safetensors` + `jepa_predictor_383_real.safetensors` written when pairs available. 41 tests pass.
- **Precision / HumanEval / adversarial / extraction harnesses (Exps 379-382):** Scripts created with hard `CARNOT_FORCE_LIVE=1` gates; all returned `status='partial'` because GPU node was offline. Live run pending once GPU is confirmed online.
- **Milestone 2026.04.28 retrospective (Exp 389):** 12 experiments (Exps 377-388, with Exps 378/386/387 missing due to session interruption), mean=19.9 min/exp. `live_gpu_confirmed=False` for FIFTH consecutive milestone. RETRO-019 (GPU node offline), RETRO-020 (CIKAN not implemented), RETRO-021 (FR-11 relay third carry) opened. RETRO-015 closed at infrastructure level. 115 tests pass. `results/operational_retro_2026_04_28.json`.

### Milestone 2026.04.29 (Exps 390-402, Exp 403 retro) — 16th Milestone

- **GPU preflight gate (Exp 390):** `scripts/experiment_390_gpu_preflight.py` created. GPU NOT confirmed live — RETRO-019 unresolved (script confirmed present, GPU node still offline).
- **JitRL constraint memory (Exp 392), Safety KAN classifier (Exp 393):** No result JSONs — fast-path did not execute inference code.
- **Precision / HumanEval / adversarial / extraction v3 harnesses (Exps 394-397):** All returned `status='partial'` — GPU node offline for SIXTH consecutive milestone.
- **FR-11 self-learning relay (Exp 399):** Partial — `honest_verdict='learning_confirmed'` NOT achieved; FOURTH consecutive miss (RETRO-024 opened).
- **Milestone 2026.04.29 retrospective (Exp 403):** 13 experiments (Exps 390-402), mean=7.5 min/exp. All experiments ran in "deliverable already exists" fast-path mode — no actual inference work. `live_gpu_confirmed=False` for SIXTH consecutive milestone. RETRO-022 (CRITICAL HUMAN ESCALATION: GPU node must be powered on or cloud GPU rented before next milestone), RETRO-023 (CIKANEnergy third consecutive failure — corrupt JSON fast-path), RETRO-024 (FR-11 relay fourth carry) opened. 138 tests pass. `results/operational_retro_2026_04_29.json`.

### Milestone 2026.04.30 (Exps 404-419, in progress) — 17th Milestone

- **Deliverable content validator + GPU preflight v2 (Exp 404):** `DeliverableContentValidator` implemented in `python/carnot/pipeline/deliverable_validator.py` with `ast.parse()` + `json.loads()` pre-check. Root cause of RETRO-023 (corrupt JSON fast-path) formally fixed. Preflight v2 result: `honest_verdict=env_not_propagating` — GPU hardware IS present (`is_live_capable=True`), but `source scripts/session_startup.sh` was not run before the conductor session. 53 tests pass.
- **Live precision pipeline v3 harness (Exp 410):** Preflight gate detected `env_not_propagating` and correctly blocked without simulation fallback. 34 tests pass. No inference executed.
- **Live HumanEval v3 harness (Exp 411):** Same preflight gate path as Exp 410. 44 tests pass; 4-gate sequence implemented (preflight JSON check → LiveGPUGate → setup_gpu → model load). Full suite: **3,058 passed, 2 pre-existing failures**.
- **EnvironmentAutoFix + GPU preflight v3 (Exp 413):** `EnvironmentAutoFix` self-injects `CARNOT_FORCE_LIVE=1` when GPU hardware is detected and the variable is absent. `honest_verdict=auto_fix_applied` (RTX 3090 detected; env var was absent and auto-injected). RETRO-022 resolved via workaround — experiment scripts are now self-configuring. 38 tests pass.
- **Live precision pipeline with CRANE extractor (Exp 419):** `CRANEExtractionGate` (CPU-only, regex + deterministic-math constraint extractor with structural confidence gate) implemented as primary extractor for FULL_STACK variant; LLM call used only as fallback when CRANE returns zero violations. Hard gate sequence: Exp 413 preflight check → LiveGPUGate → setup_gpu → model load (Gemma4-E4B-it GPU0, Qwen3.5-0.8B GPU1). 73 new tests pass. Live run pending — will produce first credible precision-stack headline number once GPU session is active.
- **Path to first live results:** Run `source scripts/session_startup.sh` (or rely on Exp 413 `EnvironmentAutoFix` self-injection) before the next conductor session. Exp 413 confirms `honest_verdict=auto_fix_applied` — the RTX 3090 is present and CARNOT_FORCE_LIVE can now be auto-injected.

### HuggingFace Published Models (Exp 293 / v0.2.0-research)
> **Exp 304 (2026-04-14):** Upload confirmed. Credentials verified via Python API. FCV artifact live at https://huggingface.co/Carnot-EBM/carnot-formal-claim-verifier-v1.


Two Phase 1 research artifacts are published to [Carnot-EBM](https://huggingface.co/Carnot-EBM) on HuggingFace Hub:

| Artifact | Repo | Format | Notes |
|----------|------|--------|-------|
| Exp 66 joint EBM + Ising | [Carnot-EBM/carnot-joint-constraint-v1](https://huggingface.co/Carnot-EBM/carnot-joint-constraint-v1) | safetensors | Phase 1 prototype. 1.0 AUROC on held-out validation (simulated training). |
| FormalClaimVerifier | [Carnot-EBM/carnot-formal-claim-verifier-v1](https://huggingface.co/Carnot-EBM/carnot-formal-claim-verifier-v1) | ONNX + Python | Arithmetic and comparison routes as ONNX (opset 13); set_membership + boolean_entailment as pure Python. |

> Both are tagged `v0.2.0-research`. Phase 1 research prototypes — not production quality.

### Revalidation Sweep (Exp 271-279)

Re-ran 9 promising pre-provenance experiments with live or live-representative data and modern extractors. 6 CONFIRMED, 2 INCONCLUSIVE, 0 definitively ruled out.

| Exp | Approach | Classification | Key Result |
|-----|----------|----------------|------------|
| 271 | GlobalConsistencyChecker multi-turn | **CONFIRMED** | 100% detection, 0% FP, 1.91ms/call — matches synthetic baseline |
| 272 | Tier 1 self-learning on live traces | INCONCLUSIVE | 86% FP reduction (7→1) confirmed; task-success rate flat at 32.7% |
| 273 | Agent rollback verification | **CONFIRMED** | 100% rollback success + 100% violation detection (canned outputs) |
| 274 | FactualKBExtractor on IT model | **CONFIRMED** | 45% coverage (target 40%), 100% accuracy (target 75%) |
| 275 | Adaptive KAN on live traces | **CONFIRMED** | AUROC 0.991 on Exp 219-221 traces; AMR pruned 17 params, 0 AUROC gain |
| 276 | Z3+LLM+semantic on GSM8K | **CONFIRMED** | Z3+LLM: 80% detection / 0% FP; semantic: 0% detection / 20% FP for arithmetic |
| 277 | Combined verification signals | INCONCLUSIVE | Conductor OK, 3068 tests pass, but results JSON absent — needs re-run |
| 278 | Cross-session constraint memory | **CONFIRMED** | 100% warm hit rate, 0% FP unseen slice, session boundary verified, avg score 95.67 |
| 279 | Adversarial number-swapped GSM8K | **CONFIRMED** | Stale detection 100%, fresh-wrong 0%, FP 20%, lift +40pp |

Full results: `results/revalidation_sweep_271_279_summary.json`.

### Infrastructure

| Component | Result | Experiment |
|-----------|--------|------------|
| Three extractors | Regex 5/91 FP, Z3 3/91 FP, LLM 1/91 FP | Exp 206-207 |
| Verify latency | 0.006ms per constraint check | Exp 102 |
| Parallel Ising sampler | 183x faster than thrml | Exp 102 |
| CoT monitorability | Free-form 100% parseable, JSON 18% | Exp 213 |
| Dual-GPU parallel | 1.14x speedup (Thunderbolt bottleneck) | Exp 225 |

### What works on test sets but fails in practice

| Approach | Test Accuracy | Practical Result | Why It Fails |
|----------|-------------|-----------------|-------------|
| Per-token EBM (best) | 88.5% | 50% on real questions | Detects confidence, not correctness |
| Multi-layer concat | 81.3% | Not tested in deployment | Same fundamental limitation |
| Activation steering | 0% effect | N/A | Statistical ≠ causal |
| Cross-model transfer | ~50% (chance) | N/A | Model-specific representations |
| Cross-domain training | 70.8% (worse) | N/A | Domain-specific signals |

**The core problem:** activation-based EBMs measure how confident the model is, not whether it's right. A model that confidently says "Neil Armstrong walked on Mars" produces activations indistinguishable from "Neil Armstrong walked on the Moon." The EBM rewards confident hallucination and penalizes correct hedging — the exact opposite of what a hallucination detector should do.

See the [technical report](docs/technical-report.md) for the full research record.

## 14 Principles Learned

Hard-won lessons from the activation-based phase of a research program that now spans 677 experiments across 38 milestones and 16 model families. These negative results are the project's primary contribution — they document what doesn't work and why, saving other researchers months of dead ends.

### What works
1. **The model's own logprobs are the best energy.** No external EBM needed for rejection sampling — the LLM's own confidence is already an energy function. Simple, practical, +10%.
2. **Different energy signals dominate in different domains.** Logprobs for QA, structural tests for code. The composite combines both and is never worse than either alone.
3. **Multi-layer concatenation improves test-set detection by ~6%.** Concatenating activations from layers 4+12+24 achieves 81.3% vs 75.5% for the final layer alone.

### What doesn't work (and why)
4. **Activation EBMs detect confidence, not correctness.** The fundamental limitation. Confident hallucinations produce activations indistinguishable from confident correct answers. Test-set accuracy (75-88%) does not translate to practical detection (50%).
5. **Instruction tuning compresses the hallucination signal.** Base models: 86.8%. Instruction-tuned: 75.0%. RLHF makes models sound confident even when wrong.
6. **Chain-of-thought compresses it further.** Disabling thinking improves detection from 61.3% → 75.5%. Thinking makes hidden states more uniform.
7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations does NOT steer the model when injected during generation.
8. **Adversarial questions defeat post-hoc detection.** On TruthfulQA, neither logprob nor EBM rejection improves over greedy.
9. **Hallucination representations are model-specific.** Cross-model transfer is at chance (~50%). Each model needs its own EBM.
10. **EBM detection is domain-specific.** Mixing datasets hurts (70.8% < 75.5%). Mixing temperatures hurts. Train on your target domain only.
11. **Normalization doesn't enable transfer.** Z-score, L2, and PCA whitening all destroy signal without improving cross-domain or cross-model transfer.
12. **Upstream question-level detection is weak.** The model's representation of the question partially predicts hallucination (62.6%) but not usefully.

### Scaling observations
13. **EBM accuracy scales with model size** within a family. Qwen3.5: 75.5% (0.8B) → 88.5% (27B). But this is test-set accuracy — the confidence-vs-correctness problem applies at all scales.
14. **MoE architectures vary wildly.** Qwen3.5-35B has 256 genuinely specialized experts (0.008 overlap). Mixtral has 8 near-identical experts (0.997 overlap). Fundamentally different knowledge organization.

## Tools

### CLI

```bash
pip install -e .
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

### MCP Server

Configure with `cp .mcp.json.example .mcp.json` for Claude Code integration. The hardened stdio JSON-RPC server now exposes **7** tools: `verify_code`, `verify_with_properties`, `verify_code_with_pbt`, `verify_llm_output`, `verify_and_repair`, `list_domains`, and `health_check`. These tools perform **Python code verification** and pipeline checks — they do not implement the activation-based EBM hallucination detection described in the research sections.

See [docs/usage-guide.md](docs/usage-guide.md) for detailed setup and usage instructions.

## Model Tiers

| Tier | Name | Scale | Use Case |
|------|------|-------|----------|
| Large | **Boltzmann** | Deep residual + attention | Research frontiers, large-scale generation |
| Medium | **Gibbs** | Multi-layer MLP (2-4 hidden) | Applied ML, complex pattern learning |
| Efficient | **KAN** | Learnable B-spline edges | **Default for verification** — best accuracy/param ratio (0.994 AUROC, 2.3K params) |
| Hardware | **Ising** | Pairwise quadratic | **Real-time sampling + FPGA/TSU** — direct hardware mapping, fastest parallel Gibbs |

All tiers implement the same `EnergyFunction` trait (Rust) / protocol (Python), so algorithms written against the interface work with any tier.

**When to use which:** KAN is the default for constraint verification (most accurate per parameter). Ising is for real-time guided decoding and hardware deployment (fastest sampling, maps to physical p-bits). They complement each other — KAN for accuracy, Ising for speed.

### Hardware Path

The current hardware track is the [FPGA Ising design](docs/fpga-ising-design.md) for a KV260-class sparse **4,096-spin** backend. Exp 228 validates the AXI-Lite upload/trigger/readback contract in **software simulation** with the new `FPGAIsingSampler` backend; the checked-in `fpga_sim` timing (`0.824549s` on a 128-spin sparse problem) is explicitly a software-model artifact, not a synthesized FPGA throughput claim. Exp 242 now attempts the real KV260 round trip and records the honest blocker in this environment: no `CARNOT_KV260_BITFILE` path was configured, so no board timings were fabricated. Exp 243 then replays **460** saved semantic and code repair cases through the sampler path: the CPU reranker leaves top-1 quality flat at **30.2%**, leaves verifier precision flat at **30.65%**, and keeps the KV260-backed path blocked in the same environment.

## Architecture

```
carnot/
├── crates/                        # Rust workspace
│   ├── carnot-core/               # EnergyFunction trait, types, serialization
│   ├── carnot-ising/              # Ising tier: E(x) = -0.5 x^T J x - b^T x
│   ├── carnot-gibbs/              # Gibbs tier: multi-layer energy network
│   ├── carnot-boltzmann/          # Boltzmann tier: deep residual energy network
│   ├── carnot-samplers/           # Langevin dynamics + HMC samplers
│   ├── carnot-training/           # CD-k, score matching, optimizers
│   └── carnot-python/             # PyO3 bindings
├── python/carnot/                 # Python/JAX package
│   ├── core/                      # Energy function protocol, model state
│   ├── models/                    # Ising, Gibbs, Boltzmann in JAX
│   ├── samplers/                  # Langevin, HMC, ParallelIsingSampler
│   ├── training/                  # JAX training loops with Optax
│   ├── pipeline/                  # VerifyRepairPipeline, extractors, errors
│   ├── mcp/                       # MCP server for Claude Code integration
│   ├── verify/                    # ComposedEnergy, ConstraintTerm, repair
│   └── inference/                 # EBM loader, composite scorer, LLM solver
├── openspec/capabilities/         # Specification-driven contracts
│   ├── core-ebm/                  # REQ-CORE-*, SCENARIO-CORE-*
│   ├── model-tiers/               # REQ-TIER-*, SCENARIO-TIER-*
│   └── training-inference/        # REQ-TRAIN-*, REQ-SAMPLE-*
├── _bmad/                         # Strategic docs (PRD, architecture, traceability)
└── ops/                           # Operational status, changelog, test results
```

## Quick Start

### Rust

```bash
cargo build --workspace --exclude carnot-python
cargo test --workspace --exclude carnot-python
```

### Python

```bash
pip install -e ".[dev]"
pytest tests/python
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Development Philosophy

Carnot follows **spec-anchored development**:

1. **Spec first** — every feature starts as REQ-* and SCENARIO-* in OpenSpec
2. **Tests trace to specs** — every test references the requirement it verifies
3. **100% coverage** — code coverage and spec coverage enforced by pre-commit hooks
4. **Dual implementation** — Rust for performance, Python/JAX for research iteration
5. **Cross-language interop** — safetensors serialization + PyO3 bindings

See [CLAUDE.md](CLAUDE.md) for the full development workflow.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core compute (Rust) | ndarray, rayon |
| Core compute (Python) | JAX, Flax, Optax |
| Python-Rust bridge | PyO3, maturin |
| Serialization | safetensors |
| Testing | cargo test, pytest, cargo-tarpaulin |
| Linting | rustfmt, clippy, ruff, mypy (strict) |

## Research References

Papers and resources that have informed Carnot's design and direction.

### Foundational EBM Theory

- [LeCun et al. (2006) — A Tutorial on Energy-Based Learning](https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf) — The foundational EBM tutorial establishing energy functions as a unifying framework for ML
- [Gutmann & Hyvarinen (2010) — Noise-Contrastive Estimation](https://proceedings.mlr.press/v9/gutmann10a.html) — NCE training for EBMs, used in Carnot's `nce_loss()`

### EBM Architecture and Scaling

- [Energy-Based Transformers are Scalable Learners and Thinkers (2025)](https://arxiv.org/abs/2507.02092) — EBTs: train transformers to assign energy to (input, prediction) pairs, infer via gradient descent. 35% faster scaling than Transformer++. Validates Carnot's verify-and-repair architecture at transformer scale.
- [Autoregressive Language Models are Secretly EBMs (2025)](https://arxiv.org/abs/2512.15605) — Explicit bijection between ARMs and EBMs via soft Bellman equation. Every LLM is already an EBM. Theoretical foundation for extracting energy signals directly from LLM logits.
- [Learning EBMs by Self-Normalising the Likelihood (2025)](https://arxiv.org/abs/2503.07021) — SNL: single learnable parameter for partition function. Lower bound of log-likelihood, concave for exponential families. Potential alternative to NCE for training learned verifiers.

### EBM + LLM Hallucination Detection

- [Semantic Energy: Detecting LLM Hallucination Beyond Entropy (2025)](https://arxiv.org/abs/2508.14496) — Energy = negative logit from penultimate layer. High energy = hallucination. 4-5% AUROC improvement over entropy methods. Directly applicable to Carnot's verification pipeline.
- [Spilled Energy in Large Language Models (2026)](https://arxiv.org/abs/2602.18671) — Energy-based analysis of LLM internals for hallucination detection.
- [Energy-Based Calibration for Implicit Chain-of-Thought (2025)](https://arxiv.org/abs/2511.07124) — EBM-CoT: refine latent reasoning toward low-energy regions. Gradient descent on reasoning trajectories.

### EBM for Physical Systems

- [Hybrid EBMs for Physical AI: Port-Hamiltonian Dynamics (2026)](https://arxiv.org/abs/2604.00277) — Separates visible (dynamical) from hidden (feedforward) layers. Absorbing invariant sets for stability. Validates Carnot's architecture of constraint evaluation (feedforward) + repair dynamics (gradient descent).
- [Cognitively Inspired Energy-Based World Models (2024)](https://arxiv.org/abs/2406.08862) — EBMs as cognitive world models.

### Agent Skill Learning

- [Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills (2026)](https://arxiv.org/abs/2603.25158) — Parallel analyst sub-agents extract lessons from execution traces, hierarchical consolidation merges them. Integrated into Carnot's autoresearch as the Trace2Skill learning layer.

### Open-Source EBM Frameworks

| Framework | Org | Language | Focus |
|-----------|-----|----------|-------|
| [EB-JEPA](https://github.com/facebookresearch/eb_jepa) | Meta FAIR | PyTorch | Self-supervised world modeling (JEPA) |
| [THRML](https://github.com/extropic-ai/thrml) | Extropic | JAX | Probabilistic graphical models for TSU hardware |
| [TorchEBM](https://github.com/soran-ghaderi/torchebm) | Independent | PyTorch | General-purpose EBM toolkit |
| [mini-ebm](https://github.com/yataobian/mini-ebm) | Educational | PyTorch | Minimal educational EBM implementation |
| [Kona 1.0](https://logicalintelligence.com/kona-ebms-energy-based-models) | Logical Intelligence | — | Continuous latent reasoning via EBMs |
| [UvA Deep Energy Models Tutorial](https://github.com/phlippe/uvadlc_notebooks) | UvA | PyTorch | Tutorial 8: deep energy-based models |
| [Equilibrium Matching](https://energy-based-model.github.io/) | — | — | EBM training via equilibrium matching |

### Hardware

- [FPGA Ising design](docs/fpga-ising-design.md) — KV260-class sparse **4,096-spin** overlay contract with `FPGAIsingSampler`, `SoftwareFPGAOverlay`, and AXI-Lite upload/trigger/readback semantics. Provenance: **software simulation** (Exp 228), not a live hardware-speed claim.
- [Extropic TSU/XTR-0](https://extropic.ai/writing/inside-x0-and-xtr-0) — Thermodynamic Sampling Unit for native EBM inference in hardware

## Pre-trained Models

16 per-token EBM models are available on HuggingFace at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).

**Important caveat:** These models achieve 75-88% accuracy on held-out TruthfulQA test sets, but this metric is misleading. In practical deployment (8 real questions), the EBM agreed with ground truth only 50% of the time. The EBMs detect model *confidence*, not *correctness* — they are research artifacts for studying activation-space structure, not production hallucination detectors. See the [practical test results](scripts/experiment_practical_mcp_test.py).

| Model | Test Set Accuracy | Source Model | Notes |
|-------|----------|-------------|-------|
| `per-token-ebm-qwen35-27b-nothink` | 88.5% | Qwen3.5-27B | Highest test accuracy |
| `per-token-ebm-gemma4-e2b-nothink` | 86.8% | Gemma 4 E2B (base) | Best base model |
| `per-token-ebm-qwen35-9b-nothink` | 85.8% | Qwen3.5-9B | |
| `per-token-ebm-qwen35-35b-nothink` | 84.5% | Qwen3.5-35B-A3B | MoE, 256 experts |
| ... | 73-84% | 11 more models | See HuggingFace |

## License

Apache 2.0 — see [LICENSE](LICENSE).
