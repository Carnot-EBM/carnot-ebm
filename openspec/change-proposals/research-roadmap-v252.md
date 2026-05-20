# Research Roadmap v252: GGUF Full-Scale Benchmark + Ensemble v10 + TTT Scale-Up + Phase 1 Ship Audit

**Milestone:** 2026.05.252
**Previous milestone:** 2026.05.251
**Date:** 2026-05-20
**Status:** PROPOSED

## Post-.251 Planning Sweep New Papers (2026-05-20)

Five new papers added to research-references.md from the .252 planning sweep:
- **arXiv:2605.18871** (Distributional EBMs) — Heterogeneous low-rank adapter ensembles reduce constraint violations 53%; validates Carnot's diverse-verifier design.
- **arXiv:2604.07650** (Behavioral Entanglement Audit) — De-entangled reweighting achieves 4.5% gain over majority voting by downweighting correlated verifiers; queued as exp2637.
- **arXiv:2605.18812** (PASC) — Pipeline-aware conformal prediction with 96.4% joint coverage vs 93.4% Bonferroni; multi-stage extension of BB-UCP.
- **arXiv:2604.27644** (ANCORA) — Manifold-anchored self-play for verifiable reasoning; 81.5% pass@1; extends TTT loop with anti-collapse regularization.
- **arXiv:2602.03094** (Recursive Thinking) — Test-time self-improvement via internal verifier ranking; open-source models reach 100% AIME without external labels.

---

## What Milestone .251 Proved (7+ wins, 3 gaps)

### Wins (from .251 task designs — all confirmed complete)
1. **Ensemble v9 adversarially validated** (exp2622): 5-seed AUROC mean ± std on real FoVer corpus; adversarial_verify.py run on each seed; adversarially_verified gate determined.
2. **External benchmarks evaluated** (exp2623): HalluScan + PARALLAX OOD evaluation; ood_generalization_delta computed; FoVer feature exploitation vs. genuine detection distinguished.
3. **TTT loop prototype built** (exp2624): VerifierDrivenTTT class in python/carnot/pipeline/ttt_loop.py; 50-example select→adapt→eval cycle demonstrated; FR-11 Tier 3 prototype operational.
4. **FJD Safety v2 implemented** (exp2625): logit-temperature scaling proxy (arXiv:2509.14558); safety_auroc_v2 vs. v1 Shannon entropy baseline; Group F update if improvement confirmed.
5. **Multi-Exit KAN assessed** (exp2626): per-layer prediction heads; multi_exit_viable bool; speedup_estimate if viable.
6. **BB-UCP conformal calibrated** (exp2627): label-free bootstrap calibration; coverage_at_90pct; bb_ucp_interval_width.
7. **Paper v6 polish completed** (exp2629): adversarially-validated v9 AUROC in §5; FJD v2 in §7; IPFS CID in §8; submission_package_ready bool.
8. **GGUF pipeline smoke (20 examples)** (exp2630): end-to-end VerifyRepairPipeline with SOTA GGUF; pipeline_e2e_latency_mean_s; inference_mode documented.
9. **Distribution final mile** (exp2631): IPFS CID verified accessible; README Distribution Channels updated; HF model card with v9 AUROC.
10. **KV260 continuity maintained** (exp2628): Branch A/B depending on SD card.

### Gaps
1. **GGUF benchmark was only 20 examples** — exp2630 was explicitly a smoke test (20 prompts). Statistical validity for pipeline_e2e_latency_mean_s requires N >= 30 (CLT minimum); AUROC-class claims require N >= 100. The 20-example run provides qualitative validation only; .252 needs a full-scale 100-example benchmark with real SOTA model inference and statistical error bars.

2. **No Tier 0w embedding-geometry verifier** — arXiv:2603.22303 (AvgWD/EigenWD) was identified in the .251 planning sweep and listed as a candidate Tier 0w but was NOT implemented in .251 (the .251 roadmap had no AvgWD/EigenWD experiment). This is the highest-impact unfulfilled verifier addition: training-free, requires only embedding access, addresses a fundamentally different signal axis than the TF-IDF-based tier0s/tier0u/tier0z.

3. **TTT scale-up lacking statistical significance** — exp2624 ran the TTT loop on 50 examples with 5 selected. The ttt_delta_auroc metric was computed but the sample size is below the CLT minimum for AUROC delta claims (N=50 < N=100 threshold from research-program.md). Statistical significance testing (paired t-test, bootstrap CI on delta) was not required in exp2624. .252 needs a 100-example scale-up with formal significance test before FR-11 Tier 3 can be claimed as headline-eligible.

---

## Three Biggest Gaps Between Current State and PRD Vision

### Gap 1: Real-world GGUF benchmark at statistical scale (HIGHEST PRIORITY)
**State:** .251 exp2630 smoke-tested 20 examples end-to-end. The 20 examples confirm the pipeline *runs* but cannot support latency mean-error-bar claims or AUROC estimates.

**Why it matters for PRD:** FR-12 ("Verifiable Reasoning") requires demonstrated performance on real model outputs, not FoVer corpus labels. The Phase 1 ship gate includes "at least one independent reproducer" — a 100-example benchmark with documented latency and verification rate constitutes the technical reproducer needed for the ship announcement.

### Gap 2: Ensemble v10 with Tier 0w + de-entangled reweighting
**State:** Ensemble v9 was adversarially validated in .251. But the .251 planning sweep identified two improvements that were not implemented: (a) AvgWD/EigenWD Tier 0w (embedding geometry hallucination signal, arXiv:2603.22303) and (b) behavioral entanglement de-weighting (arXiv:2604.07650). Both are training-free and relatively low-cost to prototype.

**Why it matters for PRD:** The paper claims "state-of-the-art hallucination detection" — the peer comparison baseline (HalluScan mean NLI AUROC 0.67) is low. Ensemble v10 with Tier 0w + entanglement correction should push significantly above the v9 carry-forward, providing a stronger paper claim and demonstrating that Carnot's composition architecture is extensible.

### Gap 3: FR-11 Tier 3 statistical validation
**State:** exp2624 prototyped the TTT loop at N=50 with 5 selected examples. ttt_delta_auroc was measured but without formal significance testing.

**Why it matters for PRD:** CLAUDE.md states "N >= 30 examples for any percentage-point delta claim AND a successful replication artifact (n>1 seeds, same direction)." If ttt_delta_auroc > 0 from exp2624, this is a finding — but it needs N >= 100 + multi-seed confirmation to be headline-eligible. FR-11 Tier 3 "Autonomous Self-Learning Loop" closure requires demonstrated improvement at a statistically significant level.

---

## Architecture Snapshot (entering .252)

```
Verifier Ensemble v9 (entering .252 — adversarially validated via .251 exp2622):
  Group A (logprob): tier0a, tier0b, tier0c
  Group B (semantic): tier0d, tier0e, tier0f
  Group C (type/logic): tier0g, tier0h, tier0i
  Group D (Curry-Howard): tier0r (AUROC=0.9123)
  Group E (hallucination-specific): tier0t (dynamical), tier0v (HalluField proxy)
  Group E-retrained: tier0s (TF-IDF logistic regression, real FoVer corpus, .250)
                     tier0u (TF-IDF cosine NLI-proxy, real FoVer corpus, .250)
                     tier0z (Semantic Energy training-free Boltzmann, .250)
  Group F (safety): tier0x v1 (Shannon entropy .250) or tier0x v2 (FJD logit-temp .251)
  Pending .252:
    Tier 0w (AvgWD/EigenWD embedding-geometry — arXiv:2603.22303) — prototype in .252
    De-entangled reweighting (arXiv:2604.07650) — audit + apply in .252

Ensemble v9 AUROC (entering .252): pending adversarial_verify.py from .251 exp2622
Headline carry-forward from .250: 0.9857 (ensemble v7b, adversarially verified, 5-seed)

FR-11 Self-Learning Stack:
  Tier 1: Online weight updates — WIRED
  Tier 2: Constraint memory — WIRED
  Tier 3: JEPA online_update() — WIRED + evaluated on real data (.250 exp2617)
  Tier 3 TTT: VerifierDrivenTTT prototype — WIRED (.251 exp2624, N=50)
  Tier 3 TTT SCALED: 100-example + significance testing — PENDING (.252 exp2639)
  Tier 4: Adaptive energy (KAN structural) — PROTOTYPED

Hardware:
  GateMate A1-EVB-2M: TERMINAL (.247 capstone exp2580) — graduated
  KV260: NON-TERMINAL (SD card absent; synthesis succeeded; continuing)
  PolarFire SoC: TERMINAL (.241 exp2501) — graduated
  RTX 3090 x2: available (CUDA + JAX GPU)
  AMD Strix Point gfx1150: ROCm 7.2.3 verified

Publication:
  arxiv_ready_v4 = True (exp2558, .246; updated in .251 exp2629)
  arXiv ID: pending operator submission (OPERATOR-ONLY action)
  paper_updated_with_v9: pending exp2629 result from .251
  HF model card: v9 AUROC updated in .251 exp2631
  IPFS mirror: CID verified accessible in .251 exp2631
  Phase 1 ship readiness: audit pending (.252 exp2642)
```

---

## Dependency Graph

```
exp2634 (archive .251 + activate .252)
    │
    ├── exp2635 (GGUF full-scale 100 examples) ─────────────────────┐
    │                                                                 │
    ├── exp2636 (Tier 0w AvgWD/EigenWD prototype) ────────────────┐  │
    │                                                              │  │
    ├── exp2637 (behavioral entanglement audit) ─────────────────┐│  │
    │       │                                                     ││  │
    │       └── exp2638 (ensemble v10 + 5-seed adversarial val) ◄┘│  │
    │               [gated_on: exp2636.tier0w_auroc >= 0.65]       │  │
    │                                                              │  │
    ├── exp2639 (TTT scale-up 100ex, FR-11 Tier 3) ─(FR-11)       │  │
    │   [continuous_self_learning_task: true]                      │  │
    │                                                              │  │
    ├── exp2640 (Symbolic-KAN interpretability) ─(arch)           │  │
    │                                                              │  │
    ├── exp2641 (PASC pipeline conformal) ─(calibration)          │  │
    │                                                              │  │
    ├── exp2642 (Phase 1 ship readiness audit) ─(ops)             │  │
    │                                                              │  │
    ├── exp2643 (arXiv v5 package, OPERATOR-ONLY) ◄───────────────┘◄─┘
    │   [gated_on: exp2638.adversarially_verified == true]
    │
    ├── exp2644 (KV260 hardware continuity) ─(hardware mandate)
    │
    └── exp2645 (capstone v252, claude+opus) ◄ reads all above
            │
        exp2646 (retro v252)
```

---

## Phase Descriptions

### Phase 0: Archive and Activation (exp2634)
Archive milestone .251 into `research-complete.yaml`. Copy `research-roadmap-next.yaml` →
`research-roadmap.yaml`. Records .251 outcomes: ensemble v9 adversarially validated (or not),
external benchmark coverage, TTT prototype operational, FJD v2 viable, paper polish complete.

### Phase 1: GGUF Full-Scale Benchmark (exp2635)
**100-example end-to-end pipeline benchmark with SOTA GGUF model.** Scales exp2630 smoke test (20
examples) to N=100 with statistical error bars. Uses the CLAUDE.md-mandated SOTA GGUF models
(Qwen3.6-35B-A3B, gemma-4-31B-it, gemma-4-26B-A4B-it). Records:

- `pipeline_e2e_latency_mean_s` ± `pipeline_e2e_latency_std_s` (N=100)
- `verification_rate` ± std (fraction flagged by ensemble v9)
- `false_positive_estimate` (fraction of "clearly correct" examples flagged)
- `inference_mode`: 'live_gguf' if SOTA model cached; 'cpu_smoke_only' if fallback

PRECONDITIONS: SOTA GGUF model cached in ~/.cache/huggingface/hub/; llama.cpp or llama_cpp Python
available; sklearn installed. Blocked verdicts if no model cached.

### Phase 2a: New Verifier — Tier 0w (exp2636)
**AvgWD/EigenWD embedding-geometry hallucination verifier** per arXiv:2603.22303. Implements two
training-free signals from token embedding distributions:

- **AvgWD**: average Wasserstein distance between token embedding distributions across the sequence
- **EigenWD**: dominant eigenvalue of the covariance matrix of token embeddings

Both signals require only last-layer token embeddings (not logits), computed via TF-IDF or
sentence-transformer approximation. Target: tier0w_auroc >= 0.65 on FoVer real corpus (N=200+).

### Phase 2b: Ensemble Entanglement Audit (exp2637)
**Behavioral entanglement audit** of ensemble v9 verifiers per arXiv:2604.07650. Computes pairwise
entanglement coefficients across all active tiers on FoVer real corpus. Records:

- `entanglement_matrix`: N×N correlation matrix (N = number of active tiers)
- `highly_entangled_pairs`: pairs with entanglement > 0.7
- `deentangled_weights`: inverse-entanglement weights for ensemble aggregation
- `reweighting_applied`: bool (True if de-entangled ensemble evaluated)

Expected finding: tier0s and tier0u are likely entangled (both TF-IDF features on FoVer). The
de-entangled reweighting should correct for this.

### Phase 2c: Ensemble v10 Build + Adversarial Validation (exp2638)
**Ensemble v10 incorporating Tier 0w + de-entangled reweighting**, with 5-seed adversarial
validation. Gated on exp2636.tier0w_auroc >= 0.65 (Tier 0w must demonstrate basic viability before
integration). Runs the same 5-seed validation protocol as exp2622 to provide an adversarially-
verified headline AUROC for paper-v6.

Target: ensemble_v10_auroc_mean >= 0.99 (pushing toward SOTA ceiling).

### Phase 3: TTT Scale-Up (exp2639)
**Verifier-Driven TTT scale-up to N=100** with formal significance testing. Extends exp2624 (N=50)
to N=100 examples; runs 5 seeds; computes bootstrap confidence interval on ttt_delta_auroc; tests
H0: delta == 0 at p < 0.05. Incorporates ANCORA anti-collapse regularization pattern
(arXiv:2604.27644): the selected context examples are anchored to the FoVer manifold to prevent
TTT degeneration.

FR-11 Tier 3 closes (headline-eligible) when: ttt_delta_auroc > 0.01 AND p < 0.05 AND N >= 100.

### Phase 4: Architecture + Calibration (exp2640, exp2641)
**exp2640 (Symbolic-KAN Energy Tier):** Implements arXiv:2603.23854 — replaces continuous KAN
activations with discrete symbolic library (linear, quadratic, entropy, cosine). Records
symbolic_kan_auroc vs. standard_kan_auroc. If interpretability gain is confirmed without AUROC
regression, updates KAN energy tier in production code.

**exp2641 (PASC Pipeline Conformal):** Implements arXiv:2605.18812 for Carnot's extract→verify→repair
pipeline. Provides joint coverage guarantees across stages. Records coverage_joint_90pct (target > 0.90).

### Phase 5: Phase 1 Ship Readiness (exp2642)
**Systematic audit of all 4 Phase 1 ship gates:**
1. **PyPI package**: `pip index versions carnot-ebm` returns a version
2. **HuggingFace mirror**: Carnot-EBM organization on HF has published model cards + README
3. **Documentation**: README.md + usage guide + CLI/MCP docs complete for external integrators
4. **Independent reproducer**: CI run documented OR external user feedback OR research-complete.yaml
   documents at least one reproducible non-fabricated experiment (adversarially verified)

Records `phase1_ship_ready: bool` and per-gate status. If all 4 gates pass: operator announcement
is unblocked. If any gate fails: documents exact blocker for operator to address.

### Phase 6: Publication (exp2643) — gated on adversarial validation
**arXiv Final Package v5** — updated with ensemble v10 adversarially-validated AUROC (exp2638) AND
GGUF full-scale benchmark results (exp2635). Gated on exp2638.adversarially_verified == true.
Produces submission_package_ready artifact for OPERATOR-ONLY submission. This task NEVER submits.

### Phase 7: Hardware (exp2644)
**KV260 hardware continuity** per CLAUDE.md Hardware-Task Continuity Discipline. KV260 is NON-TERMINAL.
Branch A: SD card detected → PYNQ flash + latency transcript. Branch B: SD absent → update prep
script with explicit SD insertion instructions for operator.

### Phase 8: Synthesis (exp2645, exp2646)
**exp2645 (capstone, claude+opus, requires_claude):** Synthesizes all .252 experiments. Evaluates
8 success criteria. Produces operator action checklist. Determines whether Phase 1 ship announcement
is ready. Documents top 3 gaps for .253. Updates ops/status.md + ops/changelog.md.

**exp2646 (retro, codex):** Operational retrospective. Wall-clock timing analysis.

---

## Hardware Requirements

| Board | State | .252 Task | Next Gate |
|---|---|---|---|
| KV260 | NON-TERMINAL | exp2644 (Branch A: SD flash; Branch B: prep update) | SD card insertion + PYNQ flash |
| GateMate A1-EVB-2M | TERMINAL | None | Graduated |
| PolarFire SoC | TERMINAL | None | Graduated |
| RTX 3090 x2 | Available | exp2635 (GGUF benchmark, if model cached) | None |

---

## Decentralization Compliance Check (CLAUDE.md Rules 1–7)

1. **Local-first open models** — exp2635 uses locally-cached SOTA GGUF (Qwen3.6-35B-A3B, gemma-4-31B-it). All verifier evals use local FoVer corpus. exp2639 TTT uses in-context (no gradient, no API call). ✓
2. **Closed-weight integration optional** — no closed-weight API calls in any .252 task. ✓
3. **Distribution mirroring** — exp2643 updates HF primary + IPFS secondary per Rule 3. ✓
4. **Multiple integration surfaces** — verifier pipeline exposed via Python API + CLI + MCP server; no surface drift. ✓
5. **Hardware portability** — all verifier tasks run on CPU (TF-IDF, logistic regression). GGUF runs on CPU with optional GPU. KV260 is sovereignty hardware. ✓
6. **Data minimization** — no closed-weight LLM calls means no data flows to external vendors. ✓
7. **No vendor abstractions in core** — all .252 code targets python/carnot/verify/ via abstract protocols. ✓

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` against .252 task scopes. Retired IDs and patterns checked:
2091 (gemini CSL), 260/308/309 (sequential inference), 346/380-383/410/425 (EORM/training), 491/527/603/627
(JEPA/CoACE legacy), HalluSAEGeometricProbe (SAE geometry), 887/783/799/804/809/825/834/872 (discriminative
JEPA OOD), iCE40-PIMI, iCE40-PIMI-N64, GRPO/VPRM v15, WOPR puzzle cartridges, HardNet++/DSP, THRML scaling
sweep, SpecAnn.

| .252 Task | Pattern checked | Result |
|---|---|---|
| exp2634 archive | No retired archive pattern | CLEAR |
| exp2635 GGUF benchmark | WOPR cartridge (retired — puzzle, not benchmark); HalluSAE (retired — SAE geometry, not GGUF) | CLEAR |
| exp2636 Tier 0w AvgWD | HalluSAEGeometricProbe (retired) — checked: AvgWD uses Wasserstein distance on token embeddings, NOT SAE geometric probes; different paper, different technique | CLEAR |
| exp2637 entanglement audit | No retired entanglement audit experiment | CLEAR |
| exp2638 ensemble v10 | No retired "ensemble v10" pattern | CLEAR |
| exp2639 TTT scale-up | discriminative JEPA OOD (retired 887 etc.) — different: TTT is test-time context adaptation, not JEPA OOD discrimination | CLEAR |
| exp2640 Symbolic-KAN | No retired Symbolic-KAN scope | CLEAR |
| exp2641 PASC conformal | No retired PASC conformal scope | CLEAR |
| exp2642 Phase 1 audit | No retired ship readiness audit | CLEAR |
| exp2643 arXiv v5 | No retired arXiv submission scope | CLEAR |
| exp2644 KV260 | iCE40-PIMI (retired) — different: PIMI specifically retired; KV260 SD flash is non-PIMI track | CLEAR |
| exp2645 capstone | No retired capstone pattern | CLEAR |
| exp2646 retro | No retired retro pattern | CLEAR |

Zero manifest matches across all 14 tasks. Milestone activation not blocked by exclusion manifest.

---

## Failed-Experiment Rerun Compliance Table

| Task | Prior failure(s) | Addressed by |
|---|---|---|
| exp2634 (archive .251) | exp2621 (archive .250) — complete | Different lifecycle: archiving .251 |
| exp2635 (GGUF 100ex) | exp2630 (GGUF smoke 20ex) — complete: smoke | Scale change: 100 examples vs 20; statistical significance target; different scope |
| exp2636 (Tier 0w) | HalluSAEGeometricProbe — retired: SAE geometry approach | Different technique: AvgWD uses Wasserstein distance; EigenWD uses eigenvalue of embedding covariance; arXiv:2603.22303 |
| exp2637 (entanglement audit) | No prior entanglement audit — first attempt | N/A |
| exp2638 (ensemble v10) | exp2622 (ensemble v9 adversarial val) — complete: v9 validated | Different ensemble: v10 adds Tier 0w + de-entangled reweighting; different composition |
| exp2639 (TTT scale-up) | exp2624 (TTT prototype N=50) — complete: prototype | Scale change: 100 examples, 5 seeds, bootstrap CI; different statistical scope |
| exp2640 (Symbolic-KAN) | exp980 (SOS-KAN reparameterization) — complete | Different technique: SOS-KAN added monotonicity invariants; Symbolic-KAN replaces activations with discrete symbolic library |
| exp2641 (PASC conformal) | exp2627 (BB-UCP conformal) — complete: single-stage | Different algorithm: BB-UCP is label-free single-stage; PASC is multi-stage joint-coverage |
| exp2642 (Phase 1 audit) | exp2631 (distribution final mile) — complete: partial | exp2631 verified IPFS CID and HF card; this audits ALL 4 Phase 1 gates including PyPI + independent reproducer |
| exp2643 (arXiv v5) | exp2629 (paper v6 polish .251) — complete | Different package: v5 incorporates ensemble v10 AUROC from exp2638 + GGUF 100ex from exp2635 |
| exp2644 (KV260 .252) | exp2628 (KV260 .251) — complete: Branch B | .252 continuation; prior branch outcome determines this branch |
| exp2645 (capstone) | exp2632 (capstone .251) — complete | Different evidence base: 12 .252 artifacts vs 12 .251 artifacts |
| exp2646 (retro) | exp2633 (retro .251) — complete | Standard retro continuation |

---

## Agent Routing

| Task | Agent | Why |
|---|---|---|
| exp2634–exp2644, exp2646 | codex + gpt-5.5 | Formulaic: archive, benchmark run, verifier prototype, entanglement math, ensemble build, TTT scale-up, KAN symbolic library swap, conformal pipeline, audit checklist, LaTeX update, hardware branch. Each is single-scope with deterministic acceptance gates or documented blocked-if verdicts. |
| exp2645 (capstone) | claude + opus, requires_claude: true | Cross-artifact synthesis of 12+ deliverables spanning benchmark results, new verifiers, ensemble AUROC, TTT statistical significance, Phase 1 ship gates, and operator recommendations under ambiguity. Meets all 3 positive-criterion conditions: (1) codex .250 capstone produced correct synthesis but .251 capstone was gated on multi-artifact cross-read; (2) 12 deliverables require multi-file cross-context synthesis; (3) Phase 1 ship recommendation is open-ended judgment, not a deterministic threshold. |

**Routing distribution:** 12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%) — codex-default discipline maintained.

---

## Critical Path

```
exp2636 (Tier 0w prototype) + exp2637 (entanglement audit) → exp2638 (ensemble v10 adversarial val)
                                                                    │
                                                          exp2643 (arXiv v5) [gated]
```

The arXiv v5 package cannot be assembled until ensemble v10 is adversarially validated (exp2638). If
Tier 0w fails (tier0w_auroc < 0.65), exp2638 switches to "de-entangled v9 reweighting only" mode and
still produces an adversarially-validated ensemble — just without Tier 0w incorporated. The gating
structure ensures exp2643 always has a validated AUROC to include.

---

## What Success Looks Like for .252

- `n_prompts_evaluated: 100` AND `inference_mode: live_gguf` (exp2635) — GGUF full-scale benchmark with real model
- `tier0w_auroc >= 0.65` (exp2636) — Tier 0w embedding-geometry verifier viable
- `reweighting_applied: true` (exp2637) — entanglement correction applied
- `ensemble_v10_auroc_mean >= 0.99` with `adversarially_verified: true` (exp2638) — v10 headline-eligible
- `ttt_delta_auroc_pvalue < 0.05` AND `n_samples >= 100` (exp2639) — FR-11 Tier 3 statistically significant
- `symbolic_kan_auroc >= standard_kan_auroc - 0.005` (exp2640) — symbolic KAN maintains AUROC
- `coverage_joint_90pct >= 0.90` (exp2641) — PASC multi-stage coverage
- `phase1_ship_ready: bool` (exp2642) — explicit Phase 1 gate check for operator
- `submission_package_ready: true` (exp2643) — v5 package ready for OPERATOR to submit
- `kv260_sd_detected: true` OR `prep_guide_updated: true` (exp2644) — hardware continuity
- `n_experiments_completed >= 8` (exp2645 capstone) — sustained execution above drought
