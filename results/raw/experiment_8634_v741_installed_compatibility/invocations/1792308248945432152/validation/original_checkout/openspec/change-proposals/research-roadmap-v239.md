# Research Roadmap v239: AUROC Phase 4 Empirical Validation + KAN Lipschitz + KV260 Board Flash + arXiv Integrity Fix + FR-11 Tier 3 JEPA

**Milestone:** 2026.05.239
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.238 — 10/12 tasks completed (KV260 synthesis_errors=0 MAJOR WIN, FR-11 Tier 2 COMPLETE, AUROC ceiling confirmed at 0.9167, capstone blocked)

---

## What .238 Proved

Milestone .238 had 10 of 12 tasks complete. Key findings:

**Major wins:**
- **KV260 synthesis_errors=0** (exp2465, claude+opus): The RTL synthesis bug was diagnosed and fixed — a commit-level fix (carnot_ising_top.v wrapper + xilinx_unisim_stubs.v LUT6 stub) already existed; exp2465 verified it. kv260_synthesis_succeeded=True. KV260 track advances to bitstream + board-flash phase.
- **PolarFire workload validated** (exp2466): polarfire_workload_validated=True via inline Python energy computation (inline_energy_ok=True). carnot_runs_on_polarfire=False still (JAX import failure despite --no-deps attempt). Inline sovereignty claim confirmed.
- **FR-11 Tier 2 COMPLETE** (exp2463): ConstraintMemoryCache implemented with SQLite backend. cross_session_retention_rate=1.0 (all session-1 violation patterns found in session-2 domain). Tier 2 self-learning is done.
- **KAN formal bound computed** (exp2467): certified_coverage=0.0, mean_local_lipschitz=39.5. KAN is highly sensitive to input perturbations — this is a PAPER-V6 FINDING (KAN achieves 0.994 AUROC but is NOT certifiably robust). A path to robustness via Lipschitz regularization is opened.
- **Paper audit completed** (exp2468): n_claims_audited=20, audit_passed=False (1 major discrepancy: exp1100 claims 100 live 35B model calls in 7.05s — physically impossible; 2 minor discrepancies). Fixes are now well-scoped.

**Gaps confirmed:**
1. **AUROC ceiling confirmed at 0.9167**: Stouffer Z-score = 0.818, Logistic regression = 0.825 — BOTH WORSE than Fisher's method (0.9167). The current 9-verifier set has an information-theoretic ceiling under all tested aggregation methods. New high-AUROC verifier needed.
2. **Tier 0n IR Conformal excluded**: exp2460 produced AUROC=0.399 (well below chance level). The CPU logprob proxy for layer-wise activation norms does NOT work with the available telemetry manifest data structure.
3. **Tier 0o Suppressed Retrieval excluded**: exp2462 produced AUROC=0.789 (below SemanticEnergy 0.810). The paraphrase-divergence proxy using logprob halves is orthogonal but insufficient.
4. **CRANE balance_ratio=1.0 is still best**: exp2464 confirmed crane_improvement=False — mixing constrained/unconstrained decoding at the verification step does not improve verdict confidence on the current telemetry set.
5. **Capstone blocked** (exp2469, blocked_gate_check_failed): The gate on exp2461.ensemble_auroc_improved_v3==true was not met. Paper-v6 results table update is carried forward.

---

## Three Biggest Gaps vs PRD Vision (entering .239)

**Gap 1: AUROC ceiling at 0.9167 — information-theoretic ceiling under Fisher/Stouffer/Logistic**
All three aggregation methods tried. The 9-verifier set (SemanticEnergy, LaaB, FregeLogic, HALT, DiffuTruth, PCIB, LogCons Z3, HierLogCons, LaaB-meta) maxes out at 0.9167 under Fisher. New verifiers with higher individual AUROCs AND orthogonal signals are needed. Two paths:
- (a) **LLM-as-Judge Tier 0p**: use a SOTA GGUF model (Qwen3.6-35B-A3B or Gemma-4-26B-A4B) as a direct hallucination judge. LLM-as-judge has shown competitive AUROC in peer work (arXiv:2604.06216 — Blending Human and LLM Expertise). This is the highest-individual-AUROC path.
- (b) **Platt/isotonic calibration on existing verifiers**: the Fisher ceiling may partly reflect uncalibrated verifier scores. Temperature scaling may unlock marginal gains.
- (c) **Phase 4 ODAR free-energy signal**: the ODAR router (exp2455, odar_routing_implemented=True) assigns routing energies per query. Does higher ODAR energy correlate with hallucination? This is both an AUROC boost candidate AND the Phase 4 empirical validation test.

**Gap 2: Paper arXiv submission blocked — exp1100 major discrepancy + Phase 4 not empirically validated**
The audit (exp2468) found 1 major discrepancy: paper claims exp1100 ran 100 live 35B model calls in 7.05s — the adversarial verifier flagged this artifact as DURATION_TOO_SHORT (impossible). Two actions needed: (a) fix the claim in paper-v6, (b) empirically demonstrate Phase 4 free-energy verifier hypothesis per operator directive to lift the arXiv hold. Both are scoped for .239.

**Gap 3: KAN robustness — certified_coverage=0.0 blocks certified deployment**
The KAN verifier (AUROC=0.994) has mean_local_lipschitz=39.5 — it amplifies input perturbations by 39.5× on average, making certified predictions impossible. arXiv:2601.18513 (LipNeXt, Jan 2026) demonstrates Lipschitz regularization at scale. Adding a spectral normalization penalty to KAN training should reduce local_lipschitz to < 5 and enable certified_coverage > 0. This is important for paper-v6's formal guarantees section.

---

## Architecture Snapshot (entering .239)

```
Tier 0 (logit-based verifiers, current ensemble):
  Tier 0g: SemanticEnergy (AUROC=0.810)
  Tier 0h: LaaB ACL 2026 (AUROC=0.854 meta-judgment)
  Tier 0i: FregeLogic Z3+neural hybrid (AUROC=0.8831)
  Tier 0j: HALT logit-space latent probe (AUROC=0.8539)
  Tier 0k: DiffuTruth (AUROC=0.588 — low weight)
  Tier 0l: PCIB (AUROC=0.802)
  Tier 0m: HierarchicalLogCons (AUROC=0.8896, baseline)
  [EXCLUDED .238] Tier 0n: IR Conformal (AUROC=0.399 — below chance)
  [EXCLUDED .238] Tier 0o: Suppressed Retrieval (AUROC=0.789 — below SemanticEnergy)
  [NEW .239] Tier 0p: LLM-as-Judge (target AUROC > 0.90)
  Conformal Ensemble (Fisher, 9 verifiers): AUROC=0.9167 (ceiling confirmed)
  Calibrated Ensemble v4 (Platt): target AUROC > 0.9167

Tier 1 (neuro-symbolic):
  NSVIF Z3 SMT extractor (online soundness/completeness tracking)
  FregeLogic Z3+neural hybrid
  VERGE SMT repair

Tier 2 (cross-session memory — COMPLETE .238):
  ConstraintMemoryCache (SQLite, cross_session_retention_rate=1.0)

Tier 3 (predictive JEPA — NEW .239):
  JEPA Predictive Verifier: predict violation from partial response
  Based on arXiv:2509.14252 (LLM-JEPA, Sep 2025)

Samplers:
  CASAL (Carnot Adaptive Sampler), Kinetic Langevin, ODAR routing

Phase 4 (active inference):
  ODAR routing energy as free-energy proxy (exp2455, implemented)
  Fast-Slow Variant theory (arXiv:2605.12484)
  Empirical validation: .239 exp2474

Hardware:
  KV260: kv260_synthesis_succeeded=True (.238) — ready for bitstream+flash
  PolarFire: inline_energy=True, carnot_runs=False — needs JAX-free build
  GateMate: TERMINAL (off mandatory roster since .237)

Paper-v6:
  Phase 1 ship gate: MET (exp2441/.236)
  Best AUROC: 0.9167
  arXiv HOLD: requires (a) paper integrity fixes + (b) Phase 4 empirical validation
```

---

## Phase Structure

### Phase 0 — Admin (1 task)

**exp2471**: Archive milestone .238 to research-complete.yaml, activate .239 by copying research-roadmap-next.yaml to research-roadmap.yaml.

### Phase 1 — AUROC Ceiling Resolution (3 tasks)

**exp2472 — LLM-as-Judge Tier 0p** (codex):
New high-AUROC verifier using SOTA GGUF model as direct hallucination judge. Per CLAUDE.md SOTA Local Models: must include at least one of `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF`. Uses llama.cpp to load cached model and generate a YES/NO hallucination score. Maps to Tier 0p verifier score for conformal ensemble.
- Preconditions: at least one SOTA GGUF cached (ls ~/.cache/huggingface/hub/models--unsloth--)
- If no GGUF cached: blocked_model_not_cached honest verdict
- Target: tier0p_auroc > 0.90 on 36 telemetry entries

**exp2473 — Calibrated Conformal Ensemble v4** (codex):
Apply Platt scaling (sigmoid calibration) and isotonic regression to each verifier's score distribution before Fisher combination. Addresses the possibility that uncalibrated per-verifier scores are the information-theoretic ceiling cause. CPU-only, pure sklearn. Evaluates: does calibration alone push past 0.9167?
- Input: existing 9 verifier score files from prior experiments
- Methods: Platt (sklearn.calibration.CalibratedClassifierCV), isotonic regression
- Output: calibrated_ensemble_auroc, best_calibration_method

**exp2474 — Phase 4 ODAR Free-Energy Empirical Validation** (codex):
The Phase 4 hypothesis is "verifier-as-free-energy" — Carnot's energy score IS the variational free energy in an active inference framework. Empirical test: does the ODAR routing energy (from exp2455's OdarRouter) correlate with hallucination ground truth on the 36 telemetry examples? If correlation > 0.4, Phase 4 is empirically grounded. This is the CRITICAL task for lifting the arXiv hold.
- Input: ODAR router energy scores from exp2455 + telemetry manifest labels
- Metrics: Pearson r(odar_energy, hallucination_label), odar_auroc
- Output: phase4_validated bool + odar_energy_auroc

### Phase 2 — Self-Learning + Research (2 tasks)

**exp2475 — FR-11 Tier 3 JEPA Predictive Verification** (codex, continuous_self_learning_task):
Implement the JEPA predictive verifier: given the first half of an LLM response, predict whether the full response will contain a constraint violation. Train on 36 telemetry entries (partial→full correlation). Leverages arXiv:2509.14252 (LLM-JEPA architecture). This is MANDATORY per research-program.md Tier 3 requirement.
- Architecture: simple MLP predictor (partial response logprob stats → violation probability)
- Training: 24 examples, eval: 12 examples
- Gate: jepa_violation_auc >= 0.55 (better than chance)

**exp2476 — KAN Lipschitz Regularization + Recertification** (codex):
Add a spectral normalization Lipschitz penalty to KAN training to reduce mean_local_lipschitz from 39.5 toward < 5. Re-evaluate certified_coverage (target: > 0.0). Per arXiv:2601.18513 (LipNeXt), constrained-Lipschitz training enables certified robustness. This addresses the exp2467 finding directly.
- Implementation: add L_lip = max(0, ||gradient||_F - L_target) penalty to KAN loss
- L_target = 5.0 (paper-v6 certification threshold)
- Metrics: new_certified_coverage, new_mean_local_lipschitz, new_kan_auroc

### Phase 3 — Hardware Continuity (2 tasks)

**exp2477 — KV260 Bitstream Pack + Board Flash** (claude+opus, requires_claude, MANDATORY):
KV260 synthesis_errors=0 confirmed in exp2465. Terminal state requires bitstream + board execution. Attempt: (a) Vivado synthesis+P&R+bitstream if available, OR (b) nextpnr-xilinx if available. Flash via PYNQ or openFPGALoader. Record kv260_bitstream_flashed=True/False.
- Preconditions: Vivado or nextpnr-xilinx available; RTL files at kv260/
- Requires claude: (1) codex failed 3× on KV260 synthesis, (2) hardware orchestration 18+ files, (3) Vivado/nextpnr tool version judgment under open-ended error conditions
- Terminal state: kv260_bitstream_flashed=True + latency_ns recorded

**exp2478 — PolarFire Carnot Full Deploy v2** (codex, MANDATORY):
exp2466 showed inline_energy_ok=True but carnot import fails (JAX dependency). Fix: (a) patch carnot's __init__.py to replace `import jax` with a numpy-backend conditional, (b) pip install modified carnot-ebm with --no-deps, (c) verify IsingModel.energy() runs. Target: carnot_runs_on_polarfire=True.
- Preconditions: SSH reachable (polarfire 'uptime')
- retire_if_same_verdict if carnot_runs_on_polarfire=False again with same root cause

### Phase 4 — Paper Fix + arXiv Prep (2 tasks)

**exp2479 — Paper Integrity Fix** (codex):
exp2468 found 3 discrepancies: (1) MAJOR: exp1100 claims 100 live 35B calls in 7.05s (physically impossible — DURATION_TOO_SHORT); (2) MINOR: exp1068 KV260 latency claim "at 64 spins" but artifact shows max_popcount=32; (3) MINOR: exp1118/1129 GRPO sample count 25 vs 47.
Fix: update paper-v6 main.tex to reflect correct values. Gate: audit_passed=True after fixes.

**exp2480 — Phase 4 Empirical Validation Summary Report** (codex):
Synthesize Phase 4 empirical evidence: exp2474 ODAR correlation + FST theory (arXiv:2605.12484) + ODAR routing implementation (exp2455) + energy-as-routing-criterion proof-of-concept. Write a structured summary document that constitutes the Phase 4 empirical validation prerequisite for lifting the arXiv hold per operator directive.
- Output: docs/research-notes/phase4-empirical-validation-report.md
- Key claim: "Carnot's energy score acts as a free-energy routing criterion consistent with ODAR active-inference framework, empirically grounded on 36 telemetry examples"

### Phase 5 — Synthesis (2 tasks)

**exp2481 — Capstone v239** (claude+opus, requires_claude, NO GATE):
Synthesize .239 outcomes: LLM-as-Judge Tier 0p AUROC, calibrated ensemble AUROC, Phase 4 empirical validation status, KAN Lipschitz result, KV260 flash status, PolarFire carnot status, paper integrity fix. Update paper-v6 results table. Assess arXiv readiness. NOT gated on AUROC improvement (explicitly removing the gate that blocked exp2445, exp2469 twice).

**exp2482 — Retro v239** (codex):
Standard milestone operational retrospective.

---

## Dependency Graph

```
exp2471 (archive)  ──────────────────────────── Phase 0
    │
    ├── exp2472 (LLM-as-Judge Tier 0p)          Phase 1a
    ├── exp2473 (Calibrated Ensemble v4)         Phase 1b
    ├── exp2474 (Phase 4 ODAR Empirical)         Phase 1c
    ├── exp2475 (FR-11 Tier 3 JEPA)              Phase 2a
    ├── exp2476 (KAN Lipschitz)                  Phase 2b
    ├── exp2477 (KV260 Bitstream+Flash)          Phase 3a
    ├── exp2478 (PolarFire Full Deploy)           Phase 3b
    ├── exp2479 (Paper Integrity Fix)            Phase 4a
    ├── exp2480 (Phase 4 Report)                 Phase 4b
    └─────────────────────────────────────────────┐
                                                   ▼
                                          exp2481 (Capstone v239) ─── exp2482 (Retro)

No hard upstream gates on exp2481 (capstone).
All Phase 1-4 tasks run in parallel from exp2471.
```

---

## Hardware Requirements

| Board | Status entering .239 | Required .239 task | .239 terminal condition |
|-------|---------------------|-------------------|------------------------|
| KV260 | synthesis_errors=0 | exp2477 (bitstream+flash) | kv260_bitstream_flashed=True |
| PolarFire | inline_energy=True, carnot import fails | exp2478 (full deploy) | carnot_runs_on_polarfire=True |
| GateMate | TERMINAL (gatemate_bitstream_flashed=True) | none | off mandatory roster |

---

## Decentralization Check (CLAUDE.md Rules 1-7)

| Rule | Status in .239 tasks |
|------|---------------------|
| 1. Local-first open models | exp2472 uses SOTA GGUF models (local llama.cpp). No closed-weight required. |
| 2. Closed frontier optional | exp2472 checks local cache first; blocked_model_not_cached if unavailable. |
| 3. Distribution mirroring | Phase 1 ship gate MET (PyPI + HF mirror). No new artifacts to mirror. |
| 4. Multiple integration surfaces | No API surface changes in .239. |
| 5. Hardware portability | exp2477 (KV260) + exp2478 (PolarFire RISC-V) advance sovereignty hardware. |
| 6. Data minimization | exp2472 local inference — no data leaves the device. |
| 7. No vendor-specific core imports | exp2476 KAN Lipschitz adds to carnot.models.kan only, no vendor SDK. |

**All 7 decentralization rules clear.**

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` before proposing any .239 task.

| Retired scope | .239 tasks checked | Result |
|---|---|---|
| GRPO/VPRM v15 | none proposed | CLEAR |
| WOPR puzzle cartridges | none proposed | CLEAR |
| HardNet++/DSP repair stack | none proposed | CLEAR |
| THRML scaling sweep | none proposed | CLEAR |
| SpecAnn Phase 3 sampler | none proposed | CLEAR |
| exp2091 (gemini CSL Tier 1) | none proposed | CLEAR |
| iCE40 PIMI | none proposed | CLEAR |
| HalluSAE Geometric Probe | none proposed | CLEAR |
| Discriminative JEPA OOD (exp887) | exp2475 uses PREDICTIVE JEPA (different scope — see prior_failures) | DOCUMENTED |

No .239 task matches a retired experiment_id without a prior_failures block.

---

## Failed-Experiment Rerun Compliance Table

| .239 Task | Prior failure referenced | Different this time |
|-----------|--------------------------|---------------------|
| exp2472 LLM-as-Judge | exp2461 (Stouffer ceiling) | Different mechanism: live LLM judge vs logprob aggregation |
| exp2473 Calibration | exp2461 + exp2448 (Fisher/Stouffer) | Different: Platt/isotonic calibration layer, not aggregation method |
| exp2474 Phase 4 ODAR | exp1745 (alpha_t blocked) | Different: ODAR routing energy correlation vs alpha_t measurement |
| exp2475 FR-11 Tier 3 | exp887 (discriminative JEPA retired) | Predictive (partial→full) not discriminative (full→classify) |
| exp2476 KAN Lipschitz | exp2467 (certified_coverage=0.0) | Add Lipschitz penalty to training; measure after regularization |
| exp2477 KV260 Flash | exp2465 (synthesis succeeded) | NEXT STEP: bitstream+flash, not synthesis rerun |
| exp2478 PolarFire | exp2466 (inline_energy=True but carnot import fails) | Fix JAX conditional import in carnot source |
| exp2479 Paper Fix | exp2468 (audit_passed=False) | FIX task (edit paper); audit was IDENTIFY task |
| exp2480 Phase 4 Report | exp1745 (alpha_t partial) | Synthesis report from ODAR evidence, not alpha_t measurement |
| exp2481 Capstone | exp2469 (blocked_gate_check_failed) | No gate — always runs after Phase 1-4 |

---

## Agent Routing Table

| Task | Agent | Model | Justification |
|------|-------|-------|---------------|
| exp2471 | codex | gpt-5.5 | Admin: mechanical YAML + research-complete.yaml update |
| exp2472 | codex | gpt-5.5 | LLM-judge: well-defined preconditions + llama.cpp invocation pattern |
| exp2473 | codex | gpt-5.5 | Calibration: pure sklearn, deterministic |
| exp2474 | codex | gpt-5.5 | ODAR correlation: load JSON + compute Pearson r |
| exp2475 | codex | gpt-5.5 | JEPA predictor: MLP training follows established pattern |
| exp2476 | codex | gpt-5.5 | KAN Lipschitz: add penalty term to existing training loop |
| exp2477 | **claude** | **opus** | Hardware: (1) codex failed KV260 3×, (2) Vivado/nextpnr multi-file orchestration, (3) open-ended tool version judgment |
| exp2478 | codex | gpt-5.5 | PolarFire: well-scoped SSH + pip + import test |
| exp2479 | codex | gpt-5.5 | Paper fix: mechanical text editing with known changes |
| exp2480 | codex | gpt-5.5 | Phase 4 report: synthesis from existing artifacts |
| exp2481 | **claude** | **opus** | Capstone: (1) codex never completed a capstone, (2) 10+ artifact cross-reads, (3) open-ended framing |
| exp2482 | codex | gpt-5.5 | Retro: templated structure |

**2/12 tasks claude (16.7%) — meets CLAUDE.md Codex-Default rule (≤2/12).**
Both claude+opus tasks meet ALL THREE positive criteria for requires_claude per CLAUDE.md.

---

## New Research References Added for .239

### Post-.238 Planning Sweep (Milestone 2026.05.239)

**arXiv:2604.06216** — "Blending Human and LLM Expertise to Detect Hallucinations and Omissions in Mental Health Chatbot Responses" (March 2026): Quantifies vanilla LLM-as-judge limitations (54% accuracy without domain expertise). Motivates the hybrid approach in exp2472: use a strong GGUF model with few-shot hallucination examples to improve judge accuracy beyond 54% baseline. Queued as Tier 0p exp2472 in .239.

**arXiv:2509.14252** — "LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures" (Sep 2025): Applies JEPA framework to LLM training, demonstrating that predictive objectives outperform standard objectives for language tasks while remaining robust to overfitting. Foundation for exp2475's FR-11 Tier 3 predictive verifier implementation.

**arXiv:2601.18513** — "LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models" (Jan 2026): Achieves certified robustness at scale via spectral normalization and Lipschitz-bounded training. Motivates exp2476's KAN Lipschitz regularization approach — applies the same spectral normalization technique to Carnot's KAN energy verifier to reduce mean_local_lipschitz from 39.5 toward < 5.
