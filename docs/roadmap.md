# Carnot Roadmap

*Auto-updated by the research conductor as experiments complete.*

## Current Milestone

**2026.04.60 (in flight): "JEPA v20 Closure + EDU PRM + Energy Prefilter"**

| # | Experiment | Status |
|---|-----------|--------|
| 780 | GPU zombie killer v12 preflight | Complete — `gpu_zombie_killer_deployed` |
| 781 | JEPA v20 data collection | Blocked — `blocked_no_live_gpu` |
| 782 | EDU PRM step selection | Complete — `edu_prm_selected_diverse` |
| 783 | JEPA v20 retrain | Complete — `jepa_v20_insufficient_data` (data volume below gate) |
| 784 | JEPA v20 cascade deploy | Blocked — `blocked_ood_auc_below_gate` |
| 785 | SOTA GGUF code repair v2 (90-min budget, 25 problems) | Blocked — `blocked_model_load_failed` |
| 786 | Gemma4 OOM fix v3 + VR threshold grid | Blocked — `blocked_no_live_gpu` |
| 787 | SSTAR energy-ranked code selection | Complete — `energy_prefilter_efficient` |
| 788 | Constraint addition from memory | Complete — `constraint_addition_zero` |
| 789 | EBM calibration alignment | Complete — `calibration_improved` |
| 790 | AMD XDNA NPU unblock v9 | Complete — `option_a_installed_no_benchmark` |
| 791+ | Upcoming (retros + Phase 4 candidates from `research-references.md`) | Queued |

## Completed Milestones

| Milestone | Theme | Experiments | Key Breakthrough |
|-----------|-------|------------|-----------------|
| 2026.04.15 | Semantic Grounding | 211-223 | +4.9pp typed IR constraints, 86% FP reduction via self-learning |
| 2026.04.16 | Scale What Works | 224-231 | +3.0pp on full 164-problem HumanEval (statistically significant) |
| 2026.04.17 | Calibrated Verification | 232-243 | Spec-grounded code repair equalizes cross-model performance |
| 2026.04.18 | Formal Claims + Predictive | 244-257 | Formal claim verifier, predictive verification gate |
| 2026.04.19 | GPU + Calibration + Publish | 258-270 | GPU acceleration stack, HuggingFace publish |
| 2026.04.20 | Revalidation Sweep | 271-280 | 6 approaches confirmed live (consistency, rollback, factual, KAN, Z3+LLM, memory) |
| 2026.04.21 | Apple Adversarial + FPGA | 281-294 | Apple adversarial benchmark, SpilledEnergy + SemanticEnergy extractors, FPGA Verilog |
| 2026.04.22 | Adversarial Completion | 294-306 | Confidence-weighted repair (86.7% FP avoidance), experiment template |
| 2026.04.23 | JEPA + Z3 + D-Wave | 307-324 | D-Wave quantum sampler, NL-to-Z3 extractor, reward hacking detection, conductor constitution |
| 2026.04.24 | GPU Benchmarks + Hardening | 325-337 | SinkProbe pre-filter (60% skip, 0% FN), CoT circuit verifier, EORM energy reward model |
| 2026.04.25 | E2E Precision + EORM | 338-350 | Three-tier pipeline (SinkProbe, EORM, Ising), constraint template library |
| 2026.04.26 | Apple Adversarial + Z3 | 351-364 | LLM-guided Z3 formalization, GPU acceleration end-to-end |
| 2026.04.27 | LLMExtractor + Self-Learning | 365-376 | CIKAN energy tier, live adversarial GSM8K |
| 2026.04.28 | Break Simulated Barrier | 377-389 | JitRL self-learning, live precision pipeline |
| 2026.04.29 | Live Results At Last | 390-403 | GPU confirmed, CIKAN, FR-11 closed |
| 2026.04.30 | Purge + First Credible Live | 404-417 | DeliverableContentValidator, env auto-fix, GPU preflight v2 |
| 2026.04.31 | EnvironmentAutoFix + VPRM | 418-424 | Env propagation workaround, VPRM architecture |
| 2026.04.32 | Live Numbers Confirmed (infrastructure) | 425-436 | Conductor timeout watchdog, DualGPU detector, FOVER annotation, Kona Phase 3 seed, provenance audit |
| 2026.04.33 | First Live Results, ThinkPRM Bridge, Boltzmann-GPT Repair | 437-449 | LongRunBenchmarkExecutor (RETRO-026 closed), Tier 2 cross-session constraint memory relay, BoltzmannRepairBridge, operational retro v7 |
| 2026.04.34 | VeriCoT Extraction, EBM-CoT Calibration, First Positive Numbers | 450-461 | **First positive verify-repair number** (Exp 451, +5pp, LIVE); Gemma4 tokenizer bug closed (RETRO-028); EBM-CoT Langevin calibration; VeriCoT/VPRM step validators |
| 2026.04.35 | Scale the First Positive — 200q Credibility, Process Hardening, FPGA Bring-Up | 462-473 | DeliverableGuard + DualGPURunner harness, session health check, live 100q precision statistical scale-up, KAEM 3.4x speedup at n=50, PPSEBM Tier 2 isolation, KV260 RTL + AXI backend (bitfile pending hardware) |
| 2026.04.36 | Fix the Root Cause — GPU VRAM Gate, Live 200q Credibility, JEPA Recovery | 474-486 | GPUVRAMGate, conductor dedup + partial-result handoff, DualGPU harness enforcement (53 scripts), batching enforcement audit, NUP Probe v1, PPSEBM real-data validation; honest negatives on JEPA quality-gate and KAEM at n=1000 |
| 2026.04.37 | Break the VRAM Deadlock — Credibility at Last, JEPA Recovery, Surprise-Driven Replay | 487-499 | **JEPA AUC recovered 0.281→0.967 via curriculum training** (Exp 492, pending live validation in Exp 510); GPUVRAMGateV2 (kill-before-check); batching pre-commit hook; GPU thermal gate; 100% retro adoption rate (first ever); KAEM at n=5000 definitively slower than MCMC across the range, FPGA-only path confirmed |
| 2026.04.38 | Break the Credibility Ceiling — Gemma4 Quantized, 100q+ Live, GPU 1 Activated | 500-512 | Gemma4 INT4 quantization unblocks RETRO-048; KAEM gaussian-mixture advantage found (RETRO-031 CLOSED); PPSEBM energy-magnitude replay (RETRO-050 CLOSED); sixth consecutive live-100q miss — RETRO-051 critical |
| 2026.04.39 | Live 100q Retry + PPSEBM Hardening | 513-524 | Live 100q JIT VRAM gate landed; PPSEBM adopted as default replay strategy; AMD XDNA NPU probe second pass (still unavailable); operational retro |
| 2026.04.40 | DualGPU Proof + Semantic Energy | 525-537 | DualGPU concurrent forward-pass validated at 13B scale (RETRO-071 downgraded from critical); Semantic Energy Tier 0d deployed; multi-tier cascade stabilised |
| 2026.04.41 | VeriCoT Closure + PRM Scale-Up | 538-549 | VeriCoT step validator scaled to 200-question live benchmark; PRM architecture refactor; JEPA-live re-run in preparation |
| 2026.04.42 | Ensemble Recall Gate + Live Re-Run | 550-562 | Ensemble recall gate fielded as preflight for VR attempts; first 200q live re-run with structured forcing; CRANE extractor re-deployed |
| 2026.04.43 | Root Cause Surgery — Execution-Based Extraction and PURE JEPA Recovery | 563-574 | CoACEExtractor FIXED via Python eval (RETRO-061 CLOSED); JEPAPUREMinForm loss implemented; KV260 FPGA bring-up v2 (bitfile pending); HalluField Tier 0e; PRA EORM beam search |
| 2026.04.44 | Recall Surgery + Contrastive JEPA — First Verified Improvement on Live Models | 575-588 | ExclusionManifest built (RETRO-056 CLOSED); JEPA v11 CPMI retrain AUC 0.444 → 1.0 via contrastive hinge (RETRO-063 CLOSED); Symbolic-KAN interpretable energy formula (MSE 0.059 vs KAEM 137.2); DSVD mid-generation hallucination detection AUC 0.976 |
| 2026.04.45 | JEPA v12 Ensemble + KAN Safety Classifier + DSVD Adapter | 589-601 | JEPA v12 ensemble with DSVD as fallback; early KAN distillation for prompt injection (pre-FPRate); DSVD adapter productionised as Tier 2.5 |
| 2026.04.46 | CoACE Recall Calibration + v13 JEPA + Memory Persistence | 602-613 | CoACE recall calibration work (still offline/live gap); JEPA v13 CAPO v1 calibration; persistent cross-session memory baselined |
| 2026.04.47 | SymCode Closed — NUP Deployed — Recall Still Blocked | 614-626 | DSVD-SymCode hybrid verifier live AUC 0.804 (RETRO-069 RESOLVED); NUP v6 Tier 0c cascade wire-in (latency 1.27 ms); 15 consecutive zero-positive VR attempts confirmed — gate closed |
| 2026.04.48 | HERMES Improved — All RETROs Carry | 627-639 | InterWhen mid-generation monitor 3× recall vs v1 (RETRO-070 partial); ORACLE FOVER v5 corpus built; HERMES tool-augmented verification adapter; JEPA v14 ORACLE-calibrated retrain v14_ood_auc=0.912 |
| 2026.04.49 | HERMES v2 Live Generation Loop + Platt JEPA + Parallel Ising Inertia | 640-651 | HERMES v2 sentence-by-sentence live generation; Ensemble Recall Gate v2 (recall 0.36, VR gate opened); JEPA v14 Platt scaling (ECE improved); OTV one-token verifier ruled out; **Exp 652 post-milestone: Prompt-Injection KAN from gpt-oss-safeguard-20b AUROC 0.9262** |
| 2026.04.50 | Prompt-Injection Safety KAN + Structured Equation Forcing + SpecGuard Mid-Generation | 653-665 | StructuredEquationForcer detection_rate_on_forced=1.0; LSEBMCL constraint memory, zero forgetting; Ising Sampler v3 RTL with h_ema register (N=64 fits XCK26 at 48.5% LUT); Live VR attempt #18 BLOCKED |
| 2026.04.51 | EnsembleGate v4 + First Positive VR + JEPA v15 | 666-677 | **RETRO-033 CLOSED** — Live VR attempt #18 v2 with structured forcing, baseline 0.36 → post 1.0 on 25 live questions (Exp 668); JEPA v15 retrain v15_ood_auc=1.0; DualGPU Proof v3 simultaneous forward-pass (partial); KV260 DFX fix |
| 2026.04.52 | VR Win Scale-Up + DualGPU Proof + JEPA Calibration | 678-691 | **RETRO-071 CLOSED** — DualGPU parallel retrain 2.02× with identical losses (Exps 684/685); exclusion manifest wired; JEPA v15 true-OOD AUC 0.475 — Exp 671 cascade eligibility retracted; VR 200q anomaly (0/200 baseline) flagged; Prompt-Injection KAN v1 with REQ-SAFE-011 invariant-guarded distillation |
| 2026.04.53 | Post-.52 Audit + Distillation Invariant + Root Cause Identification | 690-701 | Distillation invariant confirmed (teacher 6256 s); JEPA v15 `pure_loss_anti_correlation` identified as architectural failure; VR cross-model delta -1.8 pp (format-compliance artifact); publication-ready audit record produced (Exp 700) |
| 2026.04.54 | JEPA v17 RankNet + KAN v2 + FoVer v2 | 703-715 | JEPA v17 RankNet pairwise ranking loss ood_auc=0.4819 (+0.6 % vs v16, still below random); VR Gemma4 attempt #19 no-harm; Prompt-Injection KAN v2 AUROC 0.8747 (below 0.90 gate); FoVer v2 PDDL corpus 1,400 labeled pairs |
| 2026.04.55 | Step-Level Latent Probe + VarGran Gate + KAN Publication Gate | 716-730 | **Prompt-Injection KAN v3 AUROC 0.9078** — 0.90 publication gate cleared (Exp 724); **JEPA-Reasoner Probe ood_auc=1.0** latency 0.02 ms (Exp 726, new Tier 2.1 direction); **VarGran Ising skip gate** 60 % skip, FN rate 0.060 → 0.018, 7.2 % latency reduction (Exp 727); JEPA v18 LambdaRank first JEPA above random since v14 |
| 2026.04.56 | Production Deploy Cycle — Tier 2.1 + KAN Tier 0b + FR-11 Tier 1 | 731-739 | **FR-11 Tier 1 Relay fully operational** — end-to-end verified for the first time (Exp 734); Tier 2.1 JEPAReasonerProbe AUC 0.993 ± 0.005 (far above 0.75 gate); KAN Tier 0b deployed fp_rate=0.0 (Exp 735); PSV constraint specialization root cause identified + recovery confirmed; FR-11 Tier 2 cross-session memory relay functional |
| 2026.04.57 | Milestone Retrospective + FR-11 Formal Closure + Privacy Filter HIGH | 740-753 | **FR-11 formally closed** — probe AUC 0.993, relay operational, tier2 memory functional (Exp 741); **RETRO-033 definitively closed** — two independent 200q VR trials both +0.0051 pp (Exp 742); **Privacy Filter KAN v2 AUROC 1.0** on 2/3 holdout datasets (Exp 743); CoCoA Tier 0f wired (AUC 0.812); DualGPU EORM+JEPA 1.83× speedup; complete slowest-5 composition change for the first time in project history |
| 2026.04.58 | PSV Repair + HLS Fix + Live Repair + SRSA Gate | 754-766 | Manifest enforcement finally applied (RETRO-MANIFEST-ENFORCEMENT CLOSED); **PSV relapse closed** via layered SRSA gate + constraint freezing + curriculum diversity (fp_rate 0.605 → 0.334, recovery sustained); **Yosys open-source FPGA synthesis validated** 2821 LUTs/2237 DFFs, no Vivado required (Exp 758); live-GPU code repair first run; **AST verifier perfect precision/recall/F1=1.0** on 50 synthetic snippets (Exp 764, Tier 0d viable) |
| 2026.04.59 | JEPA v19 Closure + SOTA GGUF + SETS Comparison + Semantic Energy | 767-779 | JEPA v19 predictive verification trained live GPU (`jepa_v19_improving`); EBRM vs EORM competitive; Adaptive Bayesian PSV matches full-scan quality at lower cost; Jailbreak Detection KAN v1 Tier 0h deployed; **HuggingFace publishing** — 2 models published, 26 READMEs updated with corrected Carnot-EBM URL (Exp 777, after deprecated-`huggingface-cli` fix); KV260 nextpnr-xilinx synthesis (PnR fails timing — Vivado remains the closure path) |
| 2026.04.Ω (interactive-debug breakthrough) | KV260 Ising Sampler Functional on Silicon | n/a (single-session hardware debug 2026-04-22) | **RETRO-074 CLOSED after 12 consecutive PS hangs.** Root cause: dtbo missing `target=<&fpga_full>` + `resets=<&zynqmp_reset 0x74..0x77>` — PS-PL AXI isolation stayed asserted, wedging CPU3 on every read. Fix plus K26 board preset + `interconnect_aresetn` wiring + AR-channel latch + LFSR clock-gate brought first AXI r/w roundtrip on real KV260 silicon: `SPIN_COUNT = 0x20` (32 decimal, confirms N=32 build), `0xDEADBEEF` write-read roundtrip clean. |

## Breakthrough Results

Results are labeled with provenance: **LIVE** (real model inference on GPU with `CARNOT_FORCE_LIVE=1`), **SIMULATED** (synthetic benchmark or canned CI cases), **DERIVED** (post-hoc analysis of prior live artifacts), or **PLACEHOLDER** (fast-path deliverable without actual inference). Audit performed 2026-04-16 after RETRO-022 root cause was identified in the conductor's env propagation — several results that were previously unlabeled turned out to be simulated.

| Result | Value | Experiment | Provenance | Significance |
|--------|-------|------------|------------|-------------|
| HumanEval code verification | +3.0pp [+0.6, +6.1] CI | Exp 226 | LIVE | Statistically significant on 164 official problems (gemma-4-E4B-it, 1574s runtime) |
| PBT bug detection rate | 99.3% (144/145) | Exp 220 | LIVE | Property-based testing catches nearly all wrong code (Qwen+Gemma, 816s) |
| Typed IR constraints | +4.9pp (Gemma4) | Exp 221 | LIVE | Prompt-side constraint extraction works (81 cases, 459s) |
| Self-learning FP reduction | 86% (7 to 1) | Exp 223 | DERIVED | Post-hoc analysis of Exps 220/221 held-out cohorts (inherits live inputs, no new inference) |
| Global consistency checker | 100% detection, 0% FP | Exp 271 | SIMULATED | Hand-crafted consistent/contradicted chains; ~1ms latency, no model inference |
| Agent rollback | 100% success | Exp 273 | SIMULATED | 10 hand-crafted workflows, `live_mode=false` |
| Z3+LLM on GSM8K arithmetic | 80% detection, 0% FP | Exp 276 | SIMULATED | Canned CI cohort (10 cases), `live_mode=false`, 2.54s runtime |
| Adversarial semantic grounding | +40pp lift | Exp 279 | SIMULATED | Model field explicitly `"Gemma4-E4B-it (simulated)"` |
| Confidence-weighted repair | 86.7% FP avoidance | Exp 332 | PLACEHOLDER | Fast-path deliverable: `duration_s=0.0`, constant confidence scores, no inference |
| SinkProbe pre-filter | 60% skip rate, 0% FN | Exp 348 | SIMULATED | `inference_mode="simulated"`, 50 synthetic samples, 1.5s total |
| First positive verify-repair | +5pp (LIVE) | Exp 451 | LIVE | Gemma4-E4B-it live GSM8K, 50 questions, Wilson CI — first time the verify-repair loop produced a signed positive signed_improvement on live data |
| JEPA step-quality discriminator | AUC 0.967 (pending live validation) | Exp 492 | DERIVED | Curriculum training (high→low confidence ordering) on Exp 442's Z3-annotated CoT pairs recovered from an AUC 0.281 regression.  Real mechanistic fix (prevents majority-class collapse), but the eval set may share structure with the training capture — **Exp 510 (milestone 2026.04.38) re-runs the discriminator on genuinely fresh live CoT pairs**.  If AUC holds near 0.967 there, the breakthrough is real; if it collapses to 0.5-0.7, the number was leakage.  Do not cite externally until Exp 510 lands. |
| Prompt-Injection KAN v3 AUROC | 0.9078 | Exp 724 | LIVE | First passing of the 0.90 publication gate for the safety KAN line.  Distilled from `gpt-oss-safeguard-20b` with invariant-guarded teacher inference (REQ-SAFE-011 teacher_inference_duration_s floor). |
| JEPA-Reasoner Probe OOD AUC | 1.0 (latency 0.02 ms) | Exp 726 | LIVE | Pre-generative Tier 2.1 candidate.  Bypasses the extractor recall bottleneck.  Scale-up to larger corpora pending. |
| RETRO-033 definitively closed | +0.00510 pp across two independent 200q VR trials | Exp 742 | LIVE | Seeds 218 and 999 produced identical signed improvement; VR pipeline credibility established after 20 prior attempts. |
| Privacy Filter KAN v2 AUROC | 1.0 on 2/3 holdout datasets, 0.985 in-distribution | Exp 743 | LIVE | Distilled from `openai/privacy-filter` teacher (17 GB model).  Two-consecutive-cycle block on the capability resolved. |
| DualGPU EORM+JEPA speedup | 1.83x (identical final loss sequential↔parallel) | Exp 746 | LIVE | RETRO-071 validated at the retrain step; Exp 383 class exited slowest-5 after 11 consecutive appearances. |
| AST verifier on synthetic code | precision/recall/F1 = 1.0 on 50 snippets | Exp 764 | SIMULATED | Tier 0d pre-filter validated offline.  Live integration is the next step. |
| KV260 Ising sampler first AXI r/w | `SPIN_COUNT=0x20` (32), `0xDEADBEEF` write-read roundtrip | 2026-04-22 interactive debug (RETRO-074) | LIVE (hardware) | First AXI-Lite read and write returned from real KV260 silicon after 12 consecutive hangs in earlier attempts.  Unblocks hardware benchmarking of the Ising sampler against the Python reference. |

**Headline claim (honest):** Live-validated signed improvements are now three-plus-one: HumanEval +3.0pp, PBT 99.3%, Typed IR +4.9pp, and Exp 451's first positive verify-repair number on GSM8K (+5pp, 50q).  The JEPA 0.967 AUC and several simulated entries remain motivating but need live re-runs before they can be cited externally; these re-runs are explicitly scheduled into the active roadmap (Exps 502/503/504/510).

## Product Roadmap

| Tier | Products | Status |
|------|----------|--------|
| A: Ship Now | LLM output verification, code quality scorer, candidate ranker | Built |
| B: Build Next | Safety classifier (from gpt-oss-safeguard), compliance checker, multi-agent arbiter | Planned |
| C: Needs Hardware | Factual grounding gate, anomaly detector, prompt quality scorer | Phase 2 |
| D: Foundation Model | Data quality filter, synthetic data validator, test oracle | Phase 3 |

## Hardware Acceleration

| Hardware | Type | Status |
|----------|------|--------|
| D-Wave Advantage | Quantum annealing | Sampler built (Exp 320) |
| Extropic Z1 | Thermodynamic sampling | Early access 2026 |
| KV260 FPGA | Digital Ising (32 spins shipped, scalable to 4K) | **Functional on silicon 2026-04-22** — AXI r/w roundtrip verified (`SPIN_COUNT=0x20`, `0xDEADBEEF` write-read). N=32/MAX_DEGREE=8 at 60 MHz, WNS +0.18 ns. N=4K target deferred to post-benchmark scale-up |
| RTX 3090 x2 | CUDA GPU | Working |
| Vulkan compute | Universal GPU | Planned for Phase 2 |
| Intel Loihi 2 | Neuromorphic | Need INRC access |
| NTT CIM | Coherent optical (100K+ spins) | Monitor |

## Phase 3: EBM/EBT Foundation Model

The long-term vision: an open-source foundation model based on hardware-acceleratable Energy-Based Models, with functional parity to Logical Intelligence's Kona.

- Continuous energy landscapes (bridge from discrete Ising/Z3)
- Non-autoregressive reasoning (generate via energy minimization)
- Language-free verification (learn constraint structure directly)
- Open-source (Apache 2.0) and hardware-portable (Vulkan/FPGA/D-Wave/TSU)
