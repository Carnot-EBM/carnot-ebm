# Carnot

**An open-source tool that catches the mistakes LLMs confidently make up.**

Large language models generate fluent text by predicting one token at a time.
That works well for language, but it doesn't check whether the answer makes
sense. Ask an LLM for `47 + 28`, and sometimes you get *"47 + 28 = 76"* with
total confidence — the arithmetic is wrong, but nothing in the generation
process noticed.

Carnot is the second pair of eyes. It reads the LLM's answer, extracts the
specific claims it makes (arithmetic, type assertions, code behaviours,
multi-step reasoning), checks each claim against the right kind of
ground-truth, and — if anything fails — sends the violations back to the
LLM as targeted feedback for a repair pass. It works with any LLM you can
call. No fine-tuning. No access to model weights.

Rust + Python/JAX, Apache 2.0, `pip install carnot`.

Current public research record: **1,576 experiments tracked, 123
artifact-backed completed milestone records**, with checked-in result artifacts
through Exp 1452 on 2026-05-07. `research-complete.yaml` currently archives
**123** completed milestone records through 2026.04.111.

Milestone .106 delivered the critical fix to thinking-mode certificate
generation: Exp 1366 (CRANE tag-first prefix injection) reached
**certificate_parse_rate=1.0** after .105 diagnosed the `<think>` token budget
exhaustion. Exp 1367 measures DiffuTruth as a complementary non-equilibrium
hallucination signal (AUROC **0.867**; KAN r=0.961). Exp 1374 reports FR-11
continuous self-learning v3 with delta **+1.596429**, non-forgetting rate
**1.0**, and `headline_result_allowed=true`. Milestone .107 completed **13 of
14** criteria: Exp 1382 ran the full-scale 100-case FoVer certificate + semantic
repair pipeline with `certificate_parse_rate=1.0`, `repair_hint_precision=1.0`,
and `semantic_validation_pass_rate=0.59`; Exp 1388 extended FR-11 self-learning
v4 to **59 fresh-verified cases** (up from 4 in .106) with self-learning delta
**+1.791464** and `headline_result_allowed=true`; Exp 1380 produced the arXiv
v11 bundle (submission-ready archive, manual upload required); Exp 1379
completed the paper integrity audit (5/5 issues resolved). Milestone .108
completed **12 of 13** criteria: Exp 1394 deployed DVI v2 + SECL combined
calibration (ECE reduced, positive AUROC delta); Exp 1395 extended FR-11
self-learning v5 to **1,508 fresh-verified cases** with self-learning delta
**+1.449** and `headline_result_allowed=true`; Exp 1396 diagnosed and fixed the
semantic validation failures (30/30 sample failures recovered); Exp 1397 ran
the full-scale 200-case pipeline with `semantic_validation_pass_rate=1.0` but
`full_pipeline_pass_rate=0.305` (below 0.40 headline gate); GRPO v8 NGRPO
retired (no improvement, all rollouts UNKNOWN). Post-milestone GitHub issue
work added structured verdict records (`VerdictRecord` API, Exp 1408),
SessionMemory portable packs (export/import/diff CLI, Exp 1403), the
manipulable-signal dependency constraint template (Exp 1403 backport), the
NLAH conductor charter, meta-harness conductor search (Exp 1281), and a
streaming verification API (Exp 1411). Milestone .109 completed **10 of 13**
criteria: repair diagnosis found 100 executable hints, the local Qwen repair
executor was deployed but accepted **0/20** repairs, DVI v3 was blocked by
non-forgetting **0.968604 < 0.99**, EBM-CoT v3 temperature scaling preserved
AUROC **0.985375** while reducing variance, full-scale pipeline v3 stayed at
`full_pipeline_pass_rate=0.305`, Discrete SB RTL spec work fit the KV260 budget
without a hardware claim, and PRM v1 measured AUROC **0.832874** on available
step labels. Milestone .110 completed **12 of 14** criteria: DCCD
schema-constrained repair v2 recovered **20/20** prototype repair cases, MCMC
best-of-N candidate search reached **1.0** repair success on the 20-case pool,
and a gated 50-case pipeline micro-validation reached
`full_pipeline_pass_rate=0.62`, up **+0.315** from Exp 1419, but remains
non-headline because live SOTA inference/runtime evidence did not run. DVI v3
was replay-threshold calibrated and deployed with non-forgetting **1.0**;
FR-11 v6 found **0** new promoted cases, so it is non-headline. PRM v2 filled
all **478** missing local step labels and reached AUROC **0.851789**. DPO stays
reranker-only until a direct local adapter/conversion path exists. Anchored
dual-path latent repair preserved accuracy **1.0 -> 1.0** where raw descent fell
**1.0 -> 0.25**. Discrete SB RTL lint/sim was blocked because the RTL source was
missing. Milestone .111 closed below threshold at **10 of 14** criteria because
the live local SOTA GGUF runtime gate failed: Exp 1442 found the Qwen3.6-35B
and Gemma4-31B GGUF files in cache and both RTX 3090s idle, but llama.cpp could
not load `libcudart.so.12`, and the optional Gemma4-26B GGUF was absent. Exp
1443 and Exp 1445 are therefore missing gated artifacts, and Exp 1444 correctly
blocked. Non-runtime tracks advanced: Exp 1440 reduced spec-coverage metadata
debt **71 -> 0** while the full suite remained red (**101 failed**, **6
errors** in the required full-suite attempt); Exp 1441 created
`hardware/kv260/discrete_sb_256.v` and a testbench, and Exp 1451 completed
RTL lint/simulation with no hardware-execution claim; Exp 1447 repaired FR-11
growth, moving fresh-verified cases **1,508 -> 1,664** with **156** new
promotions and non-forgetting **1.0**; Exp 1448 found PRM v3 selector lift
**0.0pp** on a saturated prototype candidate pool; Exp 1449 generated **24**
LTLZinc temporal cases (**12** accepted, **12** rejected), and Exp 1450 kept
the EBT/NRGPT micro-prototype smoke-only because energy converged but decoded
quality was not measured. The current test collection reports **22,491** items;
this is a collection count, not a full-suite pass claim.

## Install and run

```bash
pip install -e ".[dev]"

# five-line verification
python -c "
from carnot.pipeline import VerifyRepairPipeline
pipeline = VerifyRepairPipeline()
print(pipeline.verify('What is 15 + 27?', '15 + 27 = 42').verified)  # True
print(pipeline.verify('What is 15 + 27?', '15 + 27 = 43').verified)  # False
"
```

GPU: `pip install carnot[cuda]` for CUDA 12. On AMD/ROCm, use
`JAX_PLATFORMS=cpu`. Rust bindings (optional): `pip install carnot[rust]`
with the Rust toolchain installed.

## How it works

Three steps. The framework ships with four energy-model tiers (KAN, Ising,
Gibbs, Boltzmann); you pick one per task or let the pipeline route
automatically.

1. **Extract the claims.** A reasoning chain gets parsed into specific
   claims: equations, type assertions, code behaviours, cited facts. For
   code, this is AST parsing; for maths, it's patterns plus a second small
   LLM pass. These become the constraints the verifier checks.
2. **Check each claim.** Each claim goes to the right verifier: equations
   to a formal solver (Z3), code to property-based tests (Hypothesis),
   inconsistencies across steps to an Energy-Based Model that scores the
   whole chain. Every check produces a single **energy** score — low means
   consistent, high means something is wrong.
3. **Repair on violation.** If total energy is high, the violations go back
   to the LLM as structured feedback and it regenerates. The loop runs
   until energy is low or a budget is spent. No violations means no
   repair — the original answer returns unchanged.

The name *Carnot* and the term "energy" come from physics. An Energy-Based
Model assigns a single number to a candidate answer — low for valid, high
for invalid. This gives two useful things at once: a scoring rule that
doesn't need per-question tuning, and a gradient we can descend on when
repairing. The same mathematics also maps onto specialised hardware (FPGA
Ising machines, thermodynamic samplers), which is why Carnot can scale
beyond CPUs. A prototype runs on a KV260 FPGA board as of April 2026.

## What Carnot doesn't do

- **Carnot never modifies the LLM.** The target model stays frozen. This is
  not fine-tuning, RLHF, or DPO — it's a verification layer that sits
  alongside the model, reads its outputs, and uses its own lightweight
  scoring networks (KAN / Ising / Gibbs / Boltzmann tiers, all much smaller
  than the LLM itself) to judge answers.
- **Carnot doesn't need model access.** The pipeline works with API-only
  LLMs (Claude, GPT, Gemini) or local ones (Qwen, Gemma, Llama). No
  requirement to introspect weights or activations.
- **Carnot doesn't hallucinate new claims.** Extraction produces the
  claims *already in the answer*; verification never inserts new assertions
  that need to be trusted.

## Headline results

Headline model benchmark numbers below are from **live GPU inference on real
public models** (Qwen 3.5, Gemma 4, Qwen3.6-35B-A3B), never from simulated
runs. Infrastructure, hardware, synthetic-pilot, and adversarial-audit rows are
labeled by artifact provenance. Every result is traceable to a checked-in
experiment artifact under `results/`.

| What we measured | Result | Source |
|---|---|---|
| Wrong-code detection on the 164-problem HumanEval benchmark | **99.3%** (144 of 145 bugs flagged + 6 beyond official tests) | Exp 226 |
| HumanEval pass-rate gain from verify-and-repair (Gemma 4 4B) | **+3.0 percentage points** (95% CI [+0.6, +6.1]) | Exp 226 |
| Typed-constraint compliance gain (Gemma 4 4B) | **+4.9 percentage points** | Exp 221 |
| Prompt-injection classifier AUROC (distilled from GPT-OSS-Safeguard-20B) | **0.9078** (first to clear the 0.90 publication gate) | Exp 724 |
| Two-GPU parallel training (DualGPURunner production deployment) | **1.979× throughput** in production pipeline | Exp 856 |
| KV260 FPGA Ising sampler, AXI r/w roundtrip on real silicon | `SPIN_COUNT=0x20`, `0xDEADBEEF` write-read verified | 2026-04-22 debug closure |
| iCE40 N=8 combinational energy oracle | **134 LUTs**, bitstream generated | Exp 859 |
| StreamingCoT hallucination detector (Tier 0g) | **AUC=1.0** at stream-time | Exp 861 |
| Constraint memory bank compression | **31.25x** reduction, AUROC=1.0 maintained | Exp 865 |
| V-JEPA Tier 3 reasoning discriminator (VJEPA v2, deployed) | **OOD AUC=0.9211** (above 0.90 publication gate) | Exp 883/884 |
| SpectralAttentionProbe hallucination detector (Tier 0h) | **AUC=1.0**, bigram Laplacian spectral entropy | Exp 885 |
| IterativeSelfRepair code repair (HumanEval 50, execute-feedback-retry) | **8% → 80%** pass rate (+72pp), cross-model energy selection accuracy 1.0 | Exp 905/906 |
| EstimationVerifier SVAMP AUC (vs FoVer baseline 0.125) | **0.90** (+0.775 signed improvement) | Exp 908 |
| Symbolic-KAN arithmetic constraint verifier | **AUC 0.9344** (+0.7136 over standard KAN; interpretable symbolic labels) | Exp 937 |
| DualGPU pipeline throughput (realistic 50q workload) | **1.96x speedup** (production-ready at scale) | Exp 932 |
| Symbolic-KAN Real FoVer data (57 pairs) | **AUC=1.0** (best discriminative result in project history) | Exp 948 |
| SpilledEnergy Tier 0 training-free hallucination detector | **AUROC=1.0**, spill_separation=0.638 | Exp 949 |
| ThinkPRM Tier 2.9 generative CoT step verification | **AUROC=0.99** vs heuristic R-PRM baseline 0.85 (+0.14) | Exp 945 |
| SC-Energy Set Consistency verification | **AUROC=0.9017** | Exp 944 |
| KV260 Ising Sampler v4 RTL (N=128, K=16, sparse E-MVL, yosys synthesis) | **27,136 LUTs** (62% of 43,500 budget); 4/4 sim checks passed | Exp 958 |
| Symbolic-KAN v2 production deployment + HuggingFace + IPFS dual distribution | **AUC=1.0** integration test; model live at huggingface.co/Carnot-EBM/symbolic-kan-v2 | Exp 968 |
| PPSEBM cross-session memory (arXiv 2512.15658) | **Plateau broken**: 9/10 sessions add templates; cluster count 20 → 83 | Exp 970 |
| KAN-MILP formal property verification (monotonicity, output range, boundary) | **3 properties verified** via MILP; 11 violations found in untrained model | Exp 972 |
| KAN-MILP monotonicity enforcement (isotonic projection fix) | **11 violations eliminated**; 1.89x inference speedup post-fix; zero violations in production model | Exp 992 |
| SC-Energy Tier 2 production wiring as OOD detector | **143 tests, 0 failures**; `SCEnergyEnergyAdapter` replaces VJEPA v2 as default Tier 2 (VJEPA retained as fallback) | Exp 1001 |
| FoVer corpus expansion (Z3 + GSM8K + SOTA labeling) | **8,829 pairs** (from 216); probe AUROC 0.5694 → **0.9899** (SOS-KAN), 0.9885 (ThinkPRM); v7 added 500 hard-negative pairs and moved k=5 AUROC **0.93035 → 0.963925** | Exps 1055/1057/1119/1169/1211 |
| SOS-KAN v3 Neural-Gram energy verifier on full corpus | **AUROC=0.9545** on 6,548-pair FoVer corpus; 0 monotonicity violations across 16,000 samples; gram matrix PSD confirmed | Exp 1072 |
| Triple Integration E2E cascade (all 4 tiers active) | **50/50 questions** ran full cascade: Tier 0a → 0b → 2 → 3; incorrect_energy > correct_energy confirmed | Exp 1073 |
| KV260 FPGA Ising sampler live hardware sampling | **24.83μs mean latency**; 70 unique spin values across 100 samples, 0 failures; energy distribution non-uniform confirmed | Exp 1068 |
| FR-11 self-learning loop with SOTA 35B model (live GPU) | **alpha_t=0.38** with Qwen3.6-35B-A3B (35B MoE, live dual-GPU inference); fr11_loop_closed=true; 100 training examples appended | Exp 1077 |
| First positive live benchmark with SOTA IT model on HumanEval (Qwen3.6-35B-A3B) | HumanEval pass@1 **0% → 36%** after Carnot correction (first-ever positive delta with a SOTA instruction-tuned model) | Exp 1079 |
| Step-level PRM dataset at scale (MCTS-based labeling, full FoVer corpus) | **7,349 step-labeled examples** generated (target was 2,000); largest PRM dataset in project history | Exp 1084 |
| SemEnergy probe v1 (logit-space energy, arXiv 2508.14496) | **AUROC=0.948**, inference 0.017 ms/example (294x faster than 5 ms target); principled information-theoretic grounding for logit-spill signal | Exp 1096 |
| WOPR cartridges — Hashi + Slitherlink | Hashi **E=0** at convergence; Slitherlink rescue shipped with canonical puzzle **E=0.0**, 24 spins, app registered, 5 tests passing | Exps 1097/1124/1125/1141 |
| Phase 1c verifier joint null-space measurement | **joint_null_space_fraction=0.0** (acceptance criterion met); max_r_correlation=0.656 — verifiers correlated, AND-composition diversity expansion required before k=15 scales | Exp 1093 |
| GSM8K VeriCoT extraction fix (equation-style CoT) | **TP rate: 0.5 → 1.0**; SOTA models write "47 + 28 = 75" not prose; added `_EQ_INLINE_RE` to vericot_validator.py; closes two-milestone TP=0 blocker | Exp 1101 |
| ThinkPRM v2 retrain on 7,349-example PRM corpus | **AUROC=0.9946** (v1 baseline 0.9885); alpha_t=0.38 on training corpus; 7,349 step-labeled examples, 300 epochs | Exp 1111 |
| RLVR + SSD integration — honest negative | **No improvement** over baseline; energy filter degenerate (all scores 0.0 from k=5 AND-composition); SSD requires non-degenerate energy gradient as input | Exp 1099 |
| Energy inversion fix — AUROC=0.9774 post-retrain | Ordering restored: correct-energy 0.689 → 1.648, incorrect-energy 0.621 → 2.096; EBRM noise-filter + SOTA corpus retrain resolved OOD distribution shift | Exp 1120 |
| SOS-KAN/k=5 production fix | k=5 ensemble AUROC **0.5547 → 0.9402** after fitting corpus normalization stats; SOS-KAN individual AUROC **0.9902** | Exp 1128 |
| GRPO + verifier rewards | v1: **24% → 28%** (+4pp); v2: **19.15% → 27.66%** (+8.51pp); v4 structural warm-up reached **16% → 26%** (+10pp); v5+TinyV regressed **-35pp**, while GRPO-VPS step-level supervision reached **70% → 94%** (+24pp) | Exps 1118/1129/1159/1208/1209 |
| GRPO-VPS full training | **80% → 95%** (+15pp) on the 200-question eval; beats the GRPO v4 +10pp floor using DualGPU Qwen3.6-35B-A3B | Exp 1220 |
| Zenil alpha_t after energy retrain | **0.38 → 0.52** on 50 live-GPU Qwen3.6-35B-A3B examples; self-learning signal improved after inversion fix | Exp 1130 |
| Lagrangian cascade v2 | Accuracy preserved (**0.0pp delta**) with **3.2%** cost savings after adding verifier-score features; v1 had -22.86pp accuracy degradation | Exp 1131 |
| HalluGuard cascade router v3 | Accuracy preserved (**0.0pp delta**) with **4.4%** cost savings; entropy and embedding-distance features flagged 90.32% of Goodfire misses | Exp 1143 |
| CCTU constrained tool-use micro-benchmark | Live Qwen3.6-35B-A3B completion rate **4% → 12%** on 25 constrained tool-use tasks; semantic constraint TP=0.88, resource TP=0.48 | Exp 1144 |
| Goodfire exemplar cascade measurement | Tier-3 k=5 caught **36/36** failure exemplars across 12 categories; standalone low-tier rates remained weak (SemEnergy=0.2222, standalone Z3=0.0833) | Exp 1132 |
| Goodfire cheap-tier calibration | Combined cheap-tier TP **36.1% → 91.7%** in Exp 1145; SECL calibration then held TP at **91.7%** while reducing FPR **0.96 → 0.21** | Exps 1145/1157 |
| PRM-BiasBench-style adversarial audit | k=5 ensemble caught **60/60** style, length, and format attacks with 0 attack FPs; SemEnergy alone caught 20/60 | Exp 1133 |
| HardNet++-style projection repair | Arithmetic projection repair hit **100%** accuracy on 20 violations at **117 us**, roughly **76,130x** faster than prompt repair | Exp 1147 |
| MetaCluster SOS-KAN compression | SOS-KAN checkpoint compressed **5.03x** with AUROC **0.9902 → 0.9718** (drop 0.0184) and energy correlation 0.9966 | Exp 1148 |
| BEAVER-lite certificate tier | Sound unsafe-mass bound matched empirical violation rate in Exp 1142 (**0.400**); Exp 1158 tightened the bound to **0.3187** vs **0.300** empirical with mock logprobs; Exp 1170 switched to live llama.cpp `logits_all` logprobs with `mock_logprobs_used=false` | Exps 1142/1158/1170 |
| Phase 3/4 sampler diagnostics | Snap validity **100%** on 10,000 proxy states; HMC classified Regime C, so blocked Gibbs replaced HMC and reached **KL=0.0231** vs Boltzmann | Exps 1154/1155/1156 |
| MARCH multi-agent claim-check loop | Blinded checker reached **TP=100%**, **FPR=0%** on 36 Goodfire exemplars + 100 FoVer correct examples, above SemEnergy 22.2% and ThinkPRM 13.9% baselines | Exp 1160 |
| FPGA sampler correctness audit — v6 pivot | **KL(FPGA ‖ Gibbs) = 3.07** (v1 parallel); v4 tuning improved **0.134 → 0.1128**; v5 regressed to **0.4469**; v6 sequential Gibbs reached **KL=0.0000** vs CPU reference | Exps 1094/1134/1149/1161 |
| KANELE SOS-KAN FPGA blueprint | LUTized compressed SOS-KAN datapath estimated **0.12 us** latency and **2,408,333x** speedup vs CPU baseline; specification only, Vivado synthesis not run | Exp 1162 |
| NRGPT energy-native Phase 3 prototype | Baseline AUROC **0.8874 → 0.9209** with one energy recurrence iteration; three iterations dipped to **0.9158**, so recurrence is positive but not monotone | Exp 1163 |
| NRGPT frozen-prefix evaluation | Non-monotonicity classified as **Type B causal-context shift**, expected for recurrent EBMs rather than an architecture failure; paper-v6 framing recorded | Exp 1251 |
| Boltzmann-GPT CD training | Random-weight bridge **AUROC=0.65 → 0.960744** after 100 CD steps on a balanced FoVer slice; forward pass verified | Exps 1226/1237/1248 |
| Phase 4 active-inference pilot | Synthetic ARC-AGI-3-style 5x5 pilot solved **10/10** with action_count_ratio **0.2534** vs greedy baseline (74.7% fewer actions); not a full ARC-AGI-3 leaderboard result | Exp 1165 |
| Phase-5 in-situ training substrate | Prototype valid-action fraction **1.0**; training loop energy decrease **67.1%**, **5/5** safety gates passed, oracle accuracy held **1.0 → 1.0** | Exps 1222/1223 |
| Phase-5 adversarial/Spera audit | Pairwise verifier conditioning maxed at **P(V_i\|V_j)=1.000**, giving effective coverage **k_eff≈1** in a k=3 in-situ ensemble; architecture revision required before scale-up | Exp 1224 |
| k=5 verifier orthogonality audit | Production k=5 ensemble cleared the correlation gate: **max r=0.4617 < 0.5**, **k_eff=1.76**; AND-composition remains viable at k=5, not at k=6/15 scale claims | Exp 1256 |
| SC-Energy seventh verifier + k=6 validation | SC-Energy verifier reached **AUROC=1.0** with low pairwise correlations in Exp 1168, but k=6 AND-compose scored **0.8973** vs k=5 **0.9240** on the validation set, so k=5 remains the default | Exps 1168/1176 |
| Diffusion of Thought inference | T=1 produced **+4pp** accuracy, but longer diffusion gave no additional accuracy and AUROC stayed **0.5** through T=125; honest verdict: diminishing returns | Exp 1171 |
| NRGPT per-token energy | Per-token AUROC **0.998199** vs batch baseline **0.887409**, with error-token localization rate **1.0** | Exp 1172 |
| BiKA SOS-KAN hardware analysis | Multiply-free BiKA estimate shows **39.6%** resource reduction and `npu_feasible`; complexity estimate only, no accuracy benchmark | Exp 1174 |
| Q11 TSS instrumentation | Continuous-EBM sign bottleneck diagnostic shipped; SC-Energy/Z3 correlation **0.5466**, vulnerability score **0.4534** | Exp 1264 |
| DiffuTruth vs Carnot FoVer baseline | DiffuTruth semantic energy AUROC **0.0816** on FoVer, while Carnot SemEnergy probe reports **0.948187**; Carnot exceeds DiffuTruth's cited FEVER paper AUROC 0.725 on this artifact | Exp 1265 |
| QuantKAN 3-bit + LUT-KAN edge path | 3-bit PTQ keeps **AUROC=0.9801**; LUT-KAN simulation gives **2.5x** speedup with 12.5 KB lookup table | Exp 1266 |
| PRIME verifier selection | Weight vector written from FoVer/process-alignment audit: SemEnergyProbe **0.4183**, k5 ensemble summary **0.2773**, CausalReasoningVerifier **0.1423**, SymCodeVerifier **0.1339**, Z3 **0.0149**, SOS-KAN **0.0131** | Exp 1272 |
| GRPO v8 PRIME/VPRM smoke | Smoke-only self-learning delta **0.83798** with `MODEL_SPECS=[]` and `headline_result_allowed=false`; no SOTA GGUF model usage | Exp 1273 |
| Certificate memory replay | Replay score **0.642857 -> 1.0** (+0.357143) with **5** memory entries and **5** skill-graph candidates | Exp 1274 |
| FSNet + SnareNet continuous repair | Raw Langevin hard-constraint satisfaction **0.0** with 5 mean violations; FSNet reaches 0 violations in 1 feasibility step; SnareNet raises soft constraint satisfaction to **0.9896** with 16 adaptive repair iterations | Exps 1275/1276 |
| Gaming-verifier defense proxy | EST score-surface audit on 50 FoVer examples reports meaning-preserving instability **0.0**, meaning-changing sensitivity **1.0**, precision proxy **1.0**, recall proxy **1.0**, and vulnerability score **0.0**; not a live adversarial LLM headline | Exp 1278 |
| WOPR game cartridges — Connect Four, Hex, Nonogram, Futoshiki, Kakuro, Masyu | Connect Four: valid board **E=0**, 10 tests; Hex: Gibbs beat random **90%**; Nonogram: valid solution **E=0**, random **E=26**; Futoshiki: solution **E=0**, random **E=42**, inequality violation **E=4**; Kakuro: valid **E=0**, invalid **E=17**; Masyu: valid **E=0**, invalid **E=3** | Exps 1175/1188/1214/1227/1279/1280 |
| Certificate grammar backend bakeoff | `llama_cpp` GBNF selected as the available local constrained-generation backend; pure-Python post-hoc validation measured **13.8 ms / 1,000** documents; no model inference run | Exp 1283 |
| DVI verifier-feedback replay | Replay acceptance score **0.642857 -> 1.0** (+0.357143) over 140 eval examples with **7** claim-level memory entries; non-headline FoVer fallback provenance | Exp 1288 |
| HardNet++ nonlinear repair | Product-of-disks nonlinear constraint benchmark reached mean violations **1.0 -> 0.0**; copy-as-decode span reuse **1.0**; delta over SnareNet **1.2207** | Exp 1291 |
| DSP feasibility channel diagnostic | Repair-help channel **AUC=0.6605**, accuracy **0.6538**, false-stop rate **0.0**, false-continue rate **0.7714**; predictive but marginal | Exp 1292 |
| Prior-failure activation audit | **13/13** prior-failure checks passed, **12** gate upstream checks passed, **0** missing prior-failure entries | Exp 1296 |
| SOTA GGUF cache preflight v2 | **2/3** mandated GGUF models cached; `cached_sota_ready=false` because `unsloth/gemma-4-26B-A4B-it-GGUF` is missing; headline SOTA certificate work remains gated | Exp 1297 |
| Skill graph promotion/demotion v2 | **7** sandboxed skill candidates written from 140 replay slices: **5** promoted, **1** demoted, **1** expired; production skills were not modified | Exp 1302 |
| QueryBandits/NGC online memory policy | Reward **-0.714286 -> 0.882143**, self-learning delta **+1.596429**, accepted violations **120 -> 7**; non-headline replay provenance | Exp 1303 |
| HardNet++/DSP conservative stop policy | Replay stop policy precision **1.0** with **70** stop and **86** continue recommendations; useful operator gate, not a learned general stop rule | Exp 1305 |
| EBT/ARM/EBM-CoT energy bridge audit v2 | Local verifier-energy alignment completed; no native EBT, ARM soft-Bellman trainer, EBM-CoT optimizer, TSU, or Kona implementation added | Exp 1306 |
| Pytest memory watchdog | Per-test RSS tracking shipped with **8,192 MB** session cumulative limit and **500 MB** per-test leak threshold; sample run passed | Exp 1178 |
| Paper v5/v6 integrity remediation + arXiv v10 bundle | **18/18** integrity issues resolved after the critical fixes; .99 v2 fixes cite the latest orthogonality/TSS/DiffuTruth/QuantKAN artifacts; arXiv v10 compiled to PDF and `results/carnot-arxiv-v10-20260504.tar.gz`, but upload remains pending | Exps 1181/1182/1183/1205/1206/1269/1270 |
| llama.cpp GPU offload for GRPO | GPU offload verified at **302 tok/s** against a 50 tok/s floor; GRPO v5 then ran on DualGPU but regressed because TinyV abstained on **62.5%** of rewards | Exps 1207/1208 |
| SC-Energy regularized k=6 audit | Overfit diagnosed and regularized, but k=6 still regressed (**0.9027** vs k=5 **0.9240**); k=6 retired from the production path | Exp 1185 |
| Diffusion of Thought redesign | EBM-diffusion redesign scored **AUROC=0.4699** on 200 eval pairs; DoT retired as a near-random verifier signal | Exp 1186 |
| Latent-GRPO energy reward | Invalid-sample masking + one-sided noise produced **0.0pp** delta (**46% → 46%**) on the 100-row FoVer proxy | Exp 1187 |
| WOPR Hex game cartridge | 7x7 Hex cartridge operational; Gibbs energy player beat random **90%** of games and tied greedy at **50%** | Exp 1188 |
| Phase 4 active-inference harder-puzzle audit | After the .92 BFS tie, .94 generated 15 synthetic 15x15 scrambled puzzles where BFS hit the **100,000-state cap on 15/15**; blocked Gibbs solved **15/15** | Exps 1189/1210 |
| Prlimit memory cap | `RLIMIT_AS=8GB` from Exp 1191 caused conductor pre-test failures on JAX-heavy collection; Exp 1203 raised the cap to **32 GiB** and unblocked the gate | Exps 1191/1203 |
| KANtize SOS-KAN 4-bit quantization | 4-bit SOS-KAN preserved **AUROC=0.990137** vs full-precision **0.990228**, with **0.038 ms/example** inference latency and edge safetensors export | Exp 1199 |
| Pre-test suite rescue | RLIMIT_AS raised **8 GiB → 32 GiB**; pre-test gate collected **21,413** tests with **472 passed**, **0 failed**, **1 skipped** in the verification slice | Exp 1203 |
| Tier 1 constraint addition v2 | Added 1 high-signal constraint; precision improved **0.478 → 0.917** and FPR dropped **0.857 → 0.071** on 50 held-out cases | Exp 1212 |
| SDPO dense-reward distillation — honest negative | Energy teacher selection was strong (**0.902**), but token coverage was only **22.06%** and the measured delta was **-19.61pp** | Exp 1213 |
| Milestone .94 status | **13/13 criteria met**; publication hold still active, but critical paper fixes, arXiv v8 bundle, GPU offload, GRPO-VPS, Phase 4 harder puzzles, FoVer v7, Tier 1, SDPO measurement, and Nonogram all produced artifacts | Exp 1215 |
| Milestone .95 backfilled status | **10/13 criteria met** from Exp 1268 backfill; GRPO-VPS full training, Phase-5 A/B/C, Boltzmann-GPT seed, Futoshiki, and retro count as met; gaming defense, GRPO v6 delta, and prior-failure automation remained not met | Exps 1216-1228/1268 |
| Milestone .96 backfilled status | **2/13 criteria met**: autofill v2 shipped and retro-96 complete; orthogonality, arXiv, GRPO v6, Phase-5-D, Kakuro, gaming defense, and redesign criteria were not met | Exps 1229-1241/1268 |
| Milestone .97/.98 status | .97 backfill reports **4/13** criteria met: Boltzmann-GPT CD, NRGPT Type-B classification, retro started, and retro complete. .98 then met **5/13** criteria: orthogonality, Q11 TSS, DiffuTruth comparison, QuantKAN 3-bit, and retro complete | Exps 1242-1267/1268 |
| Milestone .99 status | **12/14 criteria met**; publication closeout, PRIME weights, certificate memory, FSNet/SnareNet, gaming defense, and WOPR Kakuro/Masyu completed. Triggered SOTA GGUF certificates blocked; Cactus stayed gated; no headline SOTA GGUF model usage | Exps 1268-1281 |
| Extropic Z1/XTR-0 integration packet | THRML backend stub and hardware integration packet shipped; live THRML benchmark blocked because `thrml_available=false` | Exp 1150 |
| arXiv package v10 + publication hold | v10 bundle ready at `results/carnot-arxiv-v10-20260504.tar.gz`; `tectonic main.tex` compiled `docs/arxiv-paper/main.pdf` at **371 KiB**; arXiv upload remains pending | Exp 1270 |
| Milestone .99 operational closeout | **10 min** for 2 closeout items, **5 min** average; both RTX 3090s idle at 4 MB / 0% with no zombie processes; pre-flight checks were seconds-scale; **30%** savings target via DualGPU-aware lanes, cached TeX/pre-flight state, immediate gate-block artifacts, and idempotent docs reconciliation | operational_retro_2026_04_99 |
| Milestone .100 status | **5/14 criteria met**; grammar backend selection, DVI replay, HardNet++ nonlinear repair, DSP diagnostics, and retro completed. SOTA certificate work, skill promotion, energy bridge, and arXiv receipt remained blocked, gated, or missing | Exps 1282-1295 |
| Milestone .101 status | **8/13 criteria met**; activation audit, cache preflight, skill graph, online memory policy, stop policy, bridge audit, arXiv hold receipt, and retro completed. SOTA certificates and downstream headline learning remain gated | Exps 1296-1308 |
| SOTA GGUF runtime recovery | Two headline local GGUF models resolved and loaded through llama.cpp: Qwen3.6-35B-A3B on GPU0 and Gemma4-31B-it on GPU1; optional Gemma4-26B-A4B remains absent, but the headline pair is ready | Exps 1309/1310 |
| ConstraintBench/SATQuest SOTA stability | **0.90** answer stability across **40** live llama.cpp responses, PySAT verification **0.525**, cross-model disagreement **0.80**, meaningful disagreement **0.0** | Exp 1311 |
| Triggered certificates with DCCD/GBNF | Overall parse rate **0.71223** and truthfulness **0.69697** over **139** attempts; raw trigger parsed **0/40**, GBNF parsed **40/40** with **21/40** truthful, DCCD parsed **40/40** with **29/40** truthful, repaired certificates were **19/19** truthful; below the 0.75 downstream gate | Exp 1312 |
| CerCE + GRPO/VPRM v11 replay self-learning | CerCE non-forgetting **1.0** and self-learning delta **+1.596429**; GRPO/VPRM replay score **0.525 -> 0.725** and verifier-feedback token-mask score **0.975**; no large GRPO training job or new model generation was run | Exps 1315/1317 |
| HardNet++/DSP learned stop policy | Held-out replay split reached stop precision **1.0** and recall **1.0** over **36** cases; DSP feasibility AUROC **0.640625**; learned policy matched the conservative replay policy, so this is not yet a broad general stop rule | Exp 1318 |
| KAN + p-bit portability audits | KAN audit records **192** BOP, **75** NABS, **24** RM, and a **6,144-byte** LUT table with FPGA as the near-term target; p-bit packet reaches KL **0.000412** to CPU Gibbs at 6-bit DAC/reuse=4 with dual-BRAM mapping ready; no FPGA/NPU/analog hardware execution claimed | Exps 1319/1320 |
| Milestone .102 status | **11/14 criteria met**; SOTA runtime recovered and certificate extraction measured, but certificate parse rate **0.71223 < 0.75** gated semantic validators, safe-prefix Cactus, and DVI certificate-tail updates; publication remains under operator hold | Exps 1309-1322 |
| SOTA token-health recovery | Empty/one-token failure diagnosed; multi-token certificate prompt recovered with top-k/EPR available, but empty/one-token rate remains **0.40** and the full parser was not rerun | Exp 1323 |
| Dynamic certificate grammar dry run | Dynamic parse rate **1.0** vs static GBNF proxy **0.75**; compile **0.417 ms**, mask proxy **0.009 ms/token**; no SOTA inference run | Exp 1339 |
| Failure-type memory policy | Non-forgetting **1.0**, self-learning delta **+1.596429**, accepted-violation delta **-0.846154**, **35** promoted and **37** demoted replay memories; non-headline replay only | Exp 1344 |
| .104 hardware/parity audits | THRML import blocked by missing `equinox`; p-bit dual-BRAM packet v2 keeps reuse=4 KL **0.000412** to CPU Gibbs; external dependency/Kona parity claims remain disallowed | Exps 1347-1349 |
| Milestone .104 status | **9/12 criteria met**; dynamic grammar, environment gate, replay self-learning, and hardware accounting advanced, while triggered SOTA certificates and semantic validator execution remained missing/gated | Exp 1350 |
| CRANE tag-first prefix injection, full-scale 100-case pipeline | **certificate_parse_rate=1.0** on 100 FoVer cases; repair_hint_precision=1.0; semantic_validation_pass_rate=0.59; full_pipeline_pass_rate=0.29; mcs_repair_localization_rate=1.0; headline_result_allowed=true | Exp 1382 |
| FR-11 continuous self-learning v4 (DVI + replay) | **59 fresh-verified cases** (up from 4 in .106); self-learning delta **+1.791464**; non-forgetting rate **1.0**; 63 promoted, 41 demoted memory entries; headline_result_allowed=true | Exp 1388 |
| DVI discriminative verifier training v1 | AUROC delta **+0.003486** (0.3910 → 0.3945) on 7,059 FoVer training rows; checkpoint deployed; `dvi_deployed=true` | Exp 1381 |
| arXiv v11 bundle + paper integrity audit | Paper integrity audit: **5/5** issues resolved; v11 bundle compiled; submission-ready archive at `carnot-arxiv-v11-20260505.tar.gz`; manual upload required | Exps 1379/1380 |
| manipulable-signal-dependency constraint template (Issue #6) | `manipulable_signal_dependency` added to ConstraintTemplateLibrary; **114 tests** pass; `CaseMemoryTemplateWiring` wired for `manipulable_*`, `single_source_*`, `rag*` violation types | 2026-05-05 |
| Milestone .105 status | **9/12 criteria met**; thinking-mode budget exhaustion diagnosed as terminal negative evidence; hardware/parity work honest; publication hold active | Exp 1363 |
| Milestone .106 status | **11/13 criteria met**; tag-first CRANE injection resolved certificate_parse_rate=0.0 blocker; pre-test cascade SKIPs on exp1375/1376 required manual closeout | Exp 1376 retro |
| Milestone .107 status | **13/14 criteria met**; full-scale pipeline headline allowed; arXiv v11 ready; GRPO v7 JURY-RL no improvement (sole miss); publication hold lift recommended | Exp 1389 |
| DVI v2 + SECL combined calibration | DVI AUROC **0.394526 → 0.405984** (+0.011458); SECL ECE **0.561624 → 0.306922** (45.35096% reduction); deployed from 59 fresh cases and 1,770 held-out FoVer cases | Exp 1394 |
| FR-11 continuous self-learning v5 | **1,508 fresh-verified cases** via DVI v2/SECL path; self-learning delta **+1,449** vs Exp 1388; GRPO v8 cases integrated **0** | Exp 1395 |
| Full-scale pipeline v2, 200 cases | `certificate_parse_rate=1.0`, `semantic_validation_pass_rate=1.0`, `repair_hint_precision=1.0`, but `full_pipeline_pass_rate=0.305 < 0.40`, so not a headline result | Exp 1397 |
| Milestone .108 status | **12/13 criteria met**; arXiv submission not attempted, semantic validation fixed, full pipeline below headline gate, GRPO retired, BiPRM negative | Exp 1402 |
| Structured verdict records (Issue #3) | `VerdictRecord` + `calibrated_confidence_from_energy`; `verify_record()` API on both pipeline classes; **135 tests** pass | 2026-05-06 |
| SessionMemory portable packs (Issue #5) | `export_session_memory`, `import_session_memory`, `diff_session_memory_packs` CLI; JSON Schema v1; **89 tests** pass (1 skipped) | 2026-05-06 |
| Streaming verification API (Issue #7) | Async `verify_stream` iterator emits `VerdictRecord` objects in completion order; MCP event payload includes verdict events and `stream_end`; focused stream/MCP tests pass **10/10** | Exp 1411 |
| Certificate repair executor + pipeline v3 | Repair diagnosis found **100** executable STEP_REWRITE hints; local Qwen executor tested **20** cases with **0** accepted repairs; 200-case pipeline stayed at `full_pipeline_pass_rate=0.305`, so the exact rerun is retired | Exps 1413/1414/1419 |
| DVI v3 on 1,508 fresh cases | AUROC delta **+0.011842** beats the v2 delta **+0.011458**, but non-forgetting **0.968604 < 0.99**, so v3 was not deployed | Exp 1415 |
| EBM-CoT v3 temperature scaling | AUROC **0.985375** preserved; paraphrase energy variance **0.160449 → 0.102687** with best temperature **1.25** | Exp 1416 |
| EBRM latent trajectory drift smoke | Energy decreased monotonically, but accuracy fell **1.0 → 0.25** and planned support fraction was **0.0**; anchoring and dual-path decoding required | Exp 1417 |
| DPO-style verified-pair fallback | Reranker fallback measured **+99.834437pp**, but no GGUF fine-tune ran and `headline_result_allowed=false` | Exp 1420 |
| Test execution debt v1 | Focused embedding-store runtime failures fixed with **100%** line coverage on the touched module; collection clean, but full suite remains red on pre-existing execution/spec-coverage debt | Exp 1421 |
| Discrete SB KV260 RTL spec | RTL specification complete and estimated KV260 budget fits; no synthesis or board execution claim | Exp 1422 |
| Process reward model v1 | PRM v1 AUROC **0.832874**, step precision **0.380282**, recall **0.6** on **1,030** available traces; **478** promoted traces lack local labels | Exp 1423 |
| Milestone .109 status | **10/13 criteria met**; DVI v3, FR-11 v6, and full-pipeline headline gates carry forward | Exp 1424 |
| DCCD repair v2 + MCMC candidate search | DCCD schema-constrained repair accepted **20/20** prototype repairs; MCMC best-of-N reached **1.0** repair success vs **0.0** one-candidate baseline over 20 cases; no live SOTA inference headline claim | Exps 1428/1429 |
| Full-scale pipeline v4 micro-validation | 50-case gated micro run reached `full_pipeline_pass_rate=0.62`, up **+0.315** from Exp 1419, with `repair_success_rate=0.666667`; not headline-eligible until live-SOTA scale evidence exists | Exp 1431 |
| DVI v3 replay-balanced deployment | AUROC **0.405984 → 0.417826** (+0.011842), non-forgetting **1.0**, SECL ECE reduction **55.517229%**; deployed checkpoint | Exp 1432 |
| FR-11 self-learning v6 | DVI v3 active but **0** new promoted cases; cumulative fresh-verified count remains **1,508** and `headline_result_allowed=false` | Exp 1433 |
| PRM label completion v2 | Filled **478/478** missing labels, **0** remaining; PRM v2 AUROC **0.851789**, precision **0.306931**, recall **0.659574** | Exp 1434 |
| DPO headline provenance audit | Direct GGUF fine-tuning and local adapter path unsupported; Exp 1420 remains reranker-only, not a headline DPO training claim | Exp 1435 |
| Anchored dual-path latent repair | Raw latent descent lowered energy but dropped accuracy **1.0 → 0.25**; anchored dual-path repair kept accuracy **1.0 → 1.0** with off-support rate **0.0** | Exp 1436 |
| Discrete SB RTL lint/sim | Blocked because `hardware/kv260/discrete_sb_256.v` is missing; no lint, simulation, synthesis, board execution, or hardware claim | Exp 1437 |
| Milestone .110 status | **12/14 criteria met**; repair v2, DVI, PRM, DPO audit, and anchored latent repair advanced; FR-11 positive growth and RTL source carry forward | Exp 1438 |
| Spec coverage metadata cluster fix | Spec-coverage traceability debt **71 → 0**; focused checks passed, while required full-suite attempt stayed red (**101 failed**, **6 errors**) outside the metadata fix | Exp 1440 |
| Discrete SB RTL source + lint/sim | `hardware/kv260/discrete_sb_256.v` and testbench created; Verilator lint and Icarus simulation passed, with no KV260 board execution or hardware claim | Exps 1441/1451 |
| Live SOTA GGUF runtime preflight | **Blocked**: Qwen3.6-35B and Gemma4-31B GGUF files found and dual RTX 3090s idle, but llama.cpp failed on missing `libcudart.so.12`; Gemma4-26B GGUF absent | Exp 1442 |
| FR-11 continuous self-learning v7 | Fresh-verified cases **1,508 → 1,664** with **156** new promotions, non-forgetting **1.0**, and `headline_result_allowed=true`; no fresh LLM/live-SOTA inference used | Exp 1447 |
| PRM v3 online process-reward agent | Selector AUROC **1.0**, selected repair success **1.0**, but improvement over PRM v1 and raw best-of-N was **0.0pp** on a saturated prototype pool | Exp 1448 |
| LTLZinc temporal adapter | **24** finite-trace temporal cases generated (**12** accepted, **12** rejected) across always/eventually/next/until; no MiniZinc execution or DVI training claim | Exp 1449 |
| EBT/NRGPT micro-prototype audit | Energy converged over **8** FoVer traces with median **11** steps, but decoded quality evidence was absent; keep smoke-only | Exp 1450 |
| Milestone .111 status | **10/14 criteria met**, threshold not met; live-SOTA runtime gate blocked repair v3, energy reranker, and 100-case pre-scale artifacts | Exp 1452 |
| Current Python test collection | **22,491** Python tests collected; collection-only snapshot, not a full-suite pass claim | 2026-05-07 collection run |
| Local Claude/Codex usage snapshot | Codex reads the newest local `token_count` event; Claude aggregates local token usage and reads only subscription/tier metadata from credentials; free-form quota prose is ignored instead of guessed; focused regression tests pass | 2026-05-04 changelog |

Deeper analysis of these — including everything that **didn't** work and
why — is in the [technical report](docs/technical-report.md). Per-milestone
retrospectives are checked into `results/operational_retro_*.json`.

## Honest record

This project keeps a deliberate list of claims that looked good at first
and failed audit. We publish them alongside the claims that survived,
because without the negatives the positives become uninterpretable.

- **"+64 percentage points verify-and-repair with structured forcing"**
  (Exps 668 and 679) — retracted. The grader regex matched the forced
  output format (`COMPUTE: X op Y = Z`) only, so the "improvement" measured
  output-format compliance, not reasoning. A 0/200 baseline on Qwen 3.5-0.8B
  (normally 25–45% on GSM8K) was the physical-implausibility signal.
- **"Cross-dataset prompt-injection classifier AUROC 0.9585"** (Exp 691) —
  retracted. The confusion matrix at threshold 0.5 was `TP=0, FP=0, TN=N,
  FN=N` on every dataset. AUROC is a ranking score; the classifier detected
  zero injections in practice. AUROC-without-calibration turned out to be
  insufficient as a gate. The `tp_count > 0` gate we now enforce came out
  of this audit.
- **"JEPA v15 OOD AUC = 1.0"** (Exp 671) — retracted. Collapsed to 0.4751
  on a genuinely held-out GSM8K range (Exp 682). Cascade eligibility pulled.
- **"FPGA speedup figure ready for arXiv"** (Exp 1167 audit) — held. The
  draft figure mixed an estimated CPU sweep with a per-sample FPGA number and
  made the caveat less prominent than the claim. The v10/v11 bundles have
  remediated the figure/claim issues and compile, but upload/submission remains
  pending.

The audits that caught these (Exps 679, 682, 687, 691, and the 2026-05-02
paper-integrity audit attached to Exp 1167) are the
methodologically important outputs of those milestones, even though the
headline verdicts read as wins at the time. Live measurement + genuinely
held-out validation + threshold calibration now gate every safety-classifier
claim we publish.

## Where to go next

- **[Technical report](docs/technical-report.md)** — the full research arc
  through Exp 1452 across 123 archived milestone records, with a
  plain-English timeline of what we tried, what failed, what stuck.
- **[Roadmap](docs/roadmap.md)** — current milestone, upcoming milestones,
  hardware track, and Phase 3 (Kona-parity foundation-model) direction.
- **[`openspec/capabilities/`](openspec/capabilities/)** — per-capability
  specs + REQs + scenarios that each experiment traces back to.
- **[`openspec/change-proposals/`](openspec/change-proposals/)** — drafted
  proposals awaiting scheduling (safeguard dogfood, Garak integration,
  ECP5/Nexus open FPGA port, probability-calibration verifier, others).
- **[HuggingFace — Carnot-EBM](https://huggingface.co/Carnot-EBM)** —
  published model cards for the KAN Tier 0b, Step-Level JEPA Probe, and
  Ising sampler artifacts.

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

Hard-won lessons from the activation-based phase of a research program that now spans 1,576 tracked experiments through Exp 1452 across 123 archived milestone records and 16 model families. These negative results are the project's primary contribution — they document what doesn't work and why, saving other researchers months of dead ends.

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

**When to use which:** KAN is the default for constraint verification (most accurate per parameter). Exp 1199 adds an edge-deployment path by preserving SOS-KAN AUROC at **0.990137** after 4-bit quantization. Ising is for real-time guided decoding and hardware deployment (fastest sampling, maps to physical p-bits). They complement each other — KAN for accuracy, Ising for speed.

### Hardware Path

The current hardware track is the [FPGA Ising design](docs/fpga-ising-design.md) for a KV260-class sparse **4,096-spin** backend. Exp 228 validates the AXI-Lite upload/trigger/readback contract in **software simulation** with the new `FPGAIsingSampler` backend; the checked-in `fpga_sim` timing (`0.824549s` on a 128-spin sparse problem) is explicitly a software-model artifact, not a synthesized FPGA throughput claim. Exp 1161 is the latest correctness pivot: sequential Gibbs matched the CPU reference with **KL=0.0000** on N=8 and N=128/K=16 checks after v5 DC-continuous regressed to **0.4469**. Exp 1162 adds a KANELE SOS-KAN FPGA blueprint with an estimated **0.12 us** compressed-SOS-KAN datapath, but that is a specification estimate until Vivado synthesis and hardware timing close.

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
