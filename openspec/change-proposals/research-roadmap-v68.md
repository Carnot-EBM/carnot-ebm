# Research Roadmap v68 — Milestone 2026.04.68

**Milestone Title:** VJEPA Live Cascade + Gemma4 Code Repair + Spectral Probe
**CalVer:** 2026.04.68
**Experiments:** Exp 880–891 (12 experiments)
**Previous Milestone:** 2026.04.67 (Exps 868–879)

---

## What Milestone .67 Proved

Milestone .67 achieved 4/11 criteria and closed 0 retros. Key outcomes:

| Criterion | Result |
|-----------|--------|
| ExclusionManifestEnforcer deployed | ✓ (Exp 868) |
| GGUF download verified | ✗ blocked |
| Live code repair positive | ✗ blocked |
| Live cascade benchmark | ✗ blocked |
| JEPA OOD AUC > 0.65 | ✗ (Exp 872: ood_auc=0.484, svamp_auc=0.125) |
| StreamingCoT wired | ✓ (Exp 874) |
| FR-11 Tier 2 loop closed | ✓ (Exp 875: Lagrange+Compression relay) |
| VJEPA Tier 3 seed viable | ✓ (Exp 877: ood_auc > 0.55, kl_magnitude > 0) |
| iCE40 inertia 5x | ✗ (still 2x with alpha sweep) |
| HalluSAE AUC >= 0.65 | ✗ (Exp 878: auc_v2=0.45 < v1=0.61 — below_v1) |

**Three critical learnings from .67:**

1. **HalluSAE must be retired.** Exp 878 verdict was "below_v1" (AUC regressed from 0.61
   to 0.45). `retire_if_same_verdict: true` was set. HalluSAE approach (SAE geometry) does
   not discriminate hallucinations reliably. Adding to exclusion manifest.

2. **VJEPA works better than discriminative JEPA.** Exp 877 (VJEPA variational) achieved
   OOD viability (kl_magnitude > 0, OOD AUC > 0.55) — better than all 10 discriminative
   JEPA retrains which topped out at AUC=0.571. The variational KL regularization prevents
   OOD collapse. The path forward is to scale VJEPA, not fix discriminative JEPA.

3. **Code repair is blocked by GGUF downloads, not by Gemma4.** The EnvPropagationGuard
   and DualGPU runner are ready. The blocker is purely the GGUF file download (11 attempts).
   Gemma4-E4B-it is already available via HuggingFace transformers — using it directly
   sidesteps the GGUF pipeline entirely.

---

## Architecture Diagram (after .67)

```
LLM Response
    │
    ├─► Tier 0a: CarnotThinkProbe (ThinkPRM, generative CoT verify)
    ├─► Tier 0b: SpilledEnergyDetector (logit discrepancy, AUC=0.97)
    ├─► Tier 0c: NUP Probe v4 (bigram contrastive energy, AUC=1.0)
    ├─► Tier 0d: HallucinationBasinDetector (latent basin depth)
    ├─► Tier 0e: HalluField (token-path ensemble variance, AUC=0.97)
    ├─► Tier 0g: StreamingCoT [NEW .67] (PHaS trajectory, advisory)
    ├─► Tier 0h: SpectralAttentionProbe [NEW .68] (Laplacian eigenvalue, advisory)
    ├─► Tier 1: SinkProbe (attention sink concentration)
    ├─► Tier 2: JEPA / VJEPA [.68: switch to VJEPA if OOD AUC > 0.65]
    ├─► Tier 2.5: SymCodeVerifier (executable arithmetic, AUC=0.804 live)
    ├─► Tier 2.6: HermesVerifierAdapter (step-boundary feedback, candidate)
    ├─► Tier 2.7: CausalReasoningVerifier (causal entailment, recall=0.36)
    └─► Tier 3: Ising VerifyRepairPipeline (constraint satisfaction)

Self-Learning Loop (FR-11):
    Tier 3 violations → Tier 1 online weight update (Lagrange adaptive, .66)
    Tier 1 memory → Tier 2 JEPA training data (.67 loop closed)
    VJEPA predictions → Constraint addition (.68 target: Exp 888)
```

---

## Milestone .68 Phases

### Phase 0: Governance (Exp 880)

**Exp 880: Pre-flight v17** — Retire HalluSAE from exclusion manifest. Audit 7 open
RETROs. Update MILESTONE_PREREQS.md. Key action: add HalluSAE to ops/exclusion_manifest.yaml
with reason "retire_if_same_verdict triggered: below_v1 (auc_v2=0.45 < v1=0.61) per Exp 878".

Open RETROs entering .68:
- RETRO-MANIFEST-FULL-SCOPE: manifest patch not applied to conductor script
- RETRO-JEPA-OOD: JEPA OOD AUC < 0.65 (final attempt in .68 via VJEPA)
- RETRO-SVAMP-ZERO-AUC: SVAMP AUC = 0.125 after Exp 872
- RETRO-XILINX-TOOLS-UNAVAILABLE: KV260 synthesis blocked (Vivado not installed)
- RETRO-SOTA-MODEL-DOWNLOAD: code repair GGUF download fails (11 attempts)
- RETRO-HALLUSAE-AUC-BELOW-THRESHOLD: RETIRING this milestone
- RETRO-INERTIA-SWEEPS-TARGET-MISSED: 2x achieved, 5x target missed (root cause: parallel update needed)

### Phase 1: Live GPU with Gemma4 (Exp 881–882)

**Strategy:** Stop fighting GGUF downloads. Gemma4-E4B-it (google/gemma-4-E4B-it) is
available via HuggingFace transformers. It runs reliably on live GPU. Use it.

**Exp 881: Code Repair v8 — Gemma4 Live HumanEval**
Addresses RETRO-SOTA-MODEL-DOWNLOAD by taking a different path entirely.
- Use google/gemma-4-E4B-it via transformers (not llama.cpp / GGUF)
- Run 25 HumanEval problems
- Apply CodeExtractor + VerifyRepairPipeline
- Report signed_improvement, inference_mode=live_gpu
- This is the 12th attempt; different model, different loading path
- prior_failures: [exp870 blocked/exp870-gguf-download-failed]

**Exp 882: Live Cascade v7 — Gemma4 + 50 GSM8K**
First live cascade benchmark with a working instruction-tuned model.
- Gemma4-E4B-it on 50 GSM8K questions
- Full cascade Tiers 0-3 in sequence
- Measure: cascade_skip_rate, baseline_accuracy, carnot_accuracy, signed_improvement
- Report inference_mode=live_gpu

### Phase 2: VJEPA Deployment as Tier 2 (Exp 883–884)

**Strategy:** Scale VJEPA (Exp 877 viable seed) rather than fighting discriminative JEPA.
Generate synthetic step-level labels from ground-truth answers to expand corpus from
57 to 200+ pairs (per arxiv 2604.17957 data generation approach).

**Exp 883: VJEPA v2 — Expanded Corpus + Deeper Training**
- Generate synthetic labeled pairs: run Qwen3.5-0.8B on 100 GSM8K questions,
  compare intermediate steps against ground truth to label correct/incorrect
- Total corpus: 57 real FoVer + 100 synthetic GSM8K + 30 ARC + 20 SVAMP = 207 pairs
- Train VariationalJEPAPredictor for 200 epochs (vs 100 in Exp 877)
- Per-domain loss weighting (from DG-PRM, already in Exp 872 — reuse)
- Target: OOD AUC > 0.65, SVAMP AUC > 0.50
- If same verdict as Exp 877 (tier3_seed_viable but < 0.65): gate is "ood_auc > 0.60"
  (relaxed gate for deployment readiness since VJEPA is superior to discriminative JEPA)

**Exp 884: VJEPA Cascade Deploy** (GATED on Exp 883 ood_auc > 0.60)
- Replace Tier 2 JEPA with VJEPA v2 in ThreeTierPipeline
- Update architecture.md: Tier 2 = VJEPA v2 (variational, KL-regularized)
- Run final held-out OOD evaluation on 20 ARC + 10 SVAMP
- RETRO-JEPA-OOD: CLOSE if ood_auc > 0.65; PARTIAL_CLOSE if 0.60 < ood_auc < 0.65

### Phase 3: New Capabilities (Exp 885–886)

**Exp 885: Spectral Attention Probe — Tier 0h Advisory** (arxiv 2502.17598)
- Implement bigram co-occurrence Laplacian as attention proxy
- Compute spectral entropy E = -sum lambda_i * log(lambda_i + eps) over CoT steps
- Train linear probe on 50 synthetic CoT pairs (25 correct, 25 hallucinating)
- Wire as Tier 0h advisory in VerificationCertificate (flag is_spectrally_diffuse)
- Target AUC > 0.70; CPU-only (no GPU needed)
- Advisory only — does not short-circuit cascade

**Exp 886: Constrained Decoding Pre-filter** (arxiv 2508.15866)
- Implement Python AST validator that masks syntactically-invalid next tokens during generation
- Apply as pre-filter before VerifyRepairPipeline on HumanEval code generation
- The validator: for each partial code string, check if adding the candidate token produces
  a syntactically recoverable partial AST (using Python's ast.parse in fault-tolerant mode)
- Measure: CodeExtractor FP rate before vs after constrained decoding
- Expected: FP rate drops 30-50% because all generated code is syntactically valid

### Phase 4: Self-Learning Tier 3 (Exp 887–888)

**Exp 887: JEPA OOD Final Surgery via VJEPA Pretraining** (RETRO-JEPA-OOD final attempt)
- Use VariationalJEPAPredictor encoder (from Exp 883) as initialization for discriminative JEPA
- Hypothesis: VJEPA encoder learns OOD-stable representations; fine-tune as discriminative classifier
- Train on same corpus as Exp 883, evaluate OOD AUC
- prior_failures: [exp783, exp799, exp804, exp809, exp825, exp834, exp872]
- retire_if_same_verdict: true (if ood_auc <= 0.65, retire discriminative JEPA permanently)
- This is the LAST discriminative JEPA retrain attempt — VJEPA replaces it if this fails

**Exp 888: FR-11 Tier 3 Relay — VJEPA-Guided Constraint Addition**
- Wire VJEPA violation_probability into constraint addition pipeline
- When VJEPA predicts p(violation) > 0.7 AND Ising confirms violation: inject new constraint
- Run 5-session relay: 20 questions/session with VJEPA-triggered constraint addition
- Measure: constraint_addition_rate, precision_s1→s5, tier3_to_tier1_relay_confirmed
- This closes the full Tier 3 → Tier 1 self-learning loop (not yet achieved)

### Phase 5: Hardware (Exp 889–890)

**Exp 889: iCE40 PIMI v3 — Full Parallel Updates** (RETRO-INERTIA-SWEEPS-TARGET-MISSED)
- Root cause confirmed: EMA alone gives 2x; full-parallel updates are required for 15-25x
- Implement SynchronousPIMISampler: ALL spins update simultaneously using h_ema from PREVIOUS cycle
- This is an architectural change from checkerboard (Exp 876) to synchronous parallel
- EMA update: h_ema_new[i] = alpha * h_ema_old[i] + (1-alpha) * h_i(s_old)
- Spin update: s_new[i] = sign(h_ema_new[i]) with Metropolis acceptance
- Python simulation: compare parallel vs checkerboard; target 5x sweep reduction
- Verilog: implement synchronous parallel update; synthesize on iCE40
- prior_failures: [exp860 (2x only, checkerboard EMA), exp876 (2x only, alpha sweep only)]
- retire_if_same_verdict: true (if sweeps_reduction < 3, retire iCE40 PIMI research)

**Exp 890: GGUF Download v3 — CLI Approach**
- Fundamentally different from Exps 869/857 (which used Python API):
  use shell: `hf download <repo> <file> --local-dir models/`
  (huggingface_hub CLI, not Python huggingface_hub.hf_hub_download)
- Test on Qwen3.5-0.8B GGUF (small, reliably available)
- If CLI works: document as canonical download method; unblocks larger GGUF models
- prior_failures: [exp869 download_verified=False, exp857 blocked]
- retire_if_same_verdict: true (if CLI also fails, retire GGUF-based code repair)

### Phase 6: Retrospective (Exp 891)

**Exp 891: Milestone .68 Retrospective**
- Compute wall time, per-experiment avg, slowest-5, criteria met, retros closed
- Evaluate RETRO-HALLUSAE closure (it's being retired, count as closed)
- Evaluate RETRO-JEPA-OOD closure (VJEPA path)
- Evaluate RETRO-INERTIA status (parallel update)

---

## Dependency Graph

```
Exp 880 (preflight)
    │
    ├──► Exp 881 (code repair, Gemma4)
    │       │
    │       └──► Exp 882 (cascade, Gemma4) [can run in parallel with 881]
    │
    ├──► Exp 883 (VJEPA v2 train)
    │       │
    │       └──► Exp 884 (VJEPA deploy, GATED on 883 ood_auc > 0.60)
    │               │
    │               └──► Exp 888 (FR-11 Tier 3 relay, uses VJEPA)
    │
    ├──► Exp 885 (spectral probe, CPU-only, parallel with all)
    │
    ├──► Exp 886 (constrained decoding, CPU-only, parallel with all)
    │
    ├──► Exp 887 (JEPA final surgery, uses Exp 883 encoder)
    │
    ├──► Exp 889 (iCE40 PIMI v3, parallel with CPU exps)
    │
    └──► Exp 890 (GGUF CLI download, parallel with all)
         │
         └──► Exp 891 (retrospective, reads all result JSONs)
```

---

## Success Criteria

| # | Criterion | Target | Closes RETRO? |
|---|-----------|--------|---------------|
| 1 | hallusae_retired | Exp 880 adds HalluSAE to exclusion_manifest | RETRO-HALLUSAE (retirement = closure) |
| 2 | live_code_repair_positive | Exp 881 signed_improvement > 0 | RETRO-SOTA-MODEL-DOWNLOAD (partial) |
| 3 | live_cascade_benchmark | Exp 882 inference_mode=live_gpu | — |
| 4 | vjepa_ood_improved | Exp 883 ood_auc > 0.60 | — |
| 5 | vjepa_deployed | Exp 884 cascade_deployed=True | RETRO-JEPA-OOD (if ood_auc > 0.65) |
| 6 | spectral_probe_wired | Exp 885 AUC > 0.70 | — |
| 7 | constrained_decoding_fp_reduction | Exp 886 fp_rate_delta < -0.20 | — |
| 8 | jepa_ood_final_closed | Exp 887 ood_auc > 0.65 OR retired | RETRO-JEPA-OOD (if achieved) |
| 9 | fr11_tier3_relay | Exp 888 tier3_to_tier1_relay=True | — |
| 10 | pimi_5x | Exp 889 sweeps_reduction >= 5.0 | RETRO-INERTIA-SWEEPS-TARGET-MISSED |
| 11 | gguf_cli_verified | Exp 890 download_verified=True | RETRO-SOTA-MODEL-DOWNLOAD |

Target: 7+ criteria met, 3+ retros closed.

---

## Hardware Requirements

- **Exp 881, 882:** GPU required — google/gemma-4-E4B-it via transformers; CARNOT_FORCE_LIVE=1
- **Exp 883, 887, 888:** CPU only — VJEPA training on FoVer corpus (JAX_PLATFORMS=cpu)
- **Exp 884, 885, 886, 890:** CPU only
- **Exp 889:** CPU for simulation + OSS-CAD-Suite for iCE40 synthesis

---

## Failed-Experiment Rerun Compliance

Experiments that reference prior failures are compliant with CLAUDE.md's Failed-Experiment
Rerun Discipline:

- **Exp 881** (Code Repair v8): prior failures in exp870 (blocked/GGUF). Different approach:
  transformers loader instead of GGUF. retire_if_same_verdict=false (different root cause).
- **Exp 883** (VJEPA v2): prior related exp877 (tier3_seed_viable, OOD marginal). This is
  NOT the same experiment — it's scaling VJEPA, not rerunning the seed. addressed_by:
  "expanded corpus 57→207 pairs via synthetic labeling (arxiv 2604.17957)".
- **Exp 887** (JEPA Final Surgery): prior_failures list covers all 7 prior OOD retrain failures.
  addressed_by: "VJEPA encoder pretraining — fundamentally different initialization from
  random weights". retire_if_same_verdict=true. THIS IS THE LAST ATTEMPT.
- **Exp 889** (PIMI v3): prior_failures [exp860, exp876]. addressed_by: "synchronous full-parallel
  updates (not checkerboard) — root cause of 2x vs 5x gap now confirmed in arxiv 2604.17109".
  retire_if_same_verdict=true.
- **Exp 890** (GGUF CLI): prior_failures [exp869, exp857]. addressed_by: "hf CLI download
  (not Python API) — different codepath with timeout + retry". retire_if_same_verdict=true.
