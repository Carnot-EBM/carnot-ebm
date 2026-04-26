# Research Roadmap v69 — Milestone 2026.04.69

**Milestone Title:** VJEPA Streaming + SVAMP Root Cause + Self-Learning Tier 4 Seed
**CalVer:** 2026.04.69
**Experiments:** Exp 892–903 (12 experiments)
**Previous Milestone:** 2026.04.68 (Exps 880–891)

---

## What Milestone .68 Proved

Milestone .68 achieved significant breakthroughs, headlined by VJEPA ood_auc=0.9211
(well above the 0.65 target), and closed 3 retros. Key outcomes:

| Criterion | Result |
|-----------|--------|
| hallusae_retired | ✓ (Exp 880: added to exclusion manifest) |
| live_code_repair_positive | ✓ (Exp 881: Gemma4 via transformers loader) |
| live_cascade_benchmark | ✓ (Exp 882: inference_mode=live_gpu) |
| vjepa_ood_improved (ood_auc > 0.60) | ✓ (Exp 883: ood_auc=0.9211 — massive breakthrough) |
| vjepa_deployed (cascade_deployed=True) | ✓ (Exp 884: VJEPA Tier 2 deployed) |
| spectral_probe_viable (AUC > 0.70) | ✓ (Exp 885: Tier 0h wired) |
| constrained_decoding_fp_reduction | ✓ (Exp 886: AST pre-filter reduces FP rate) |
| jepa_ood_final_closed OR retired | ✓ (Exp 887: retired — VJEPA replaced discriminative JEPA) |
| fr11_tier3_relay | ✓ (Exp 888: VJEPA-guided constraint addition wired) |
| pimi_5x OR pimi_retired | ✗ (Exp 889: parallel updates improved but <5x or retired) |
| gguf_cli_verified OR gguf_retired | ✓ (Exp 890: resolved) |

**Three critical learnings from .68:**

1. **VJEPA is the right Tier 2 architecture — definitively.** ood_auc=0.9211 vs all prior
   discriminative JEPA attempts (best was 0.571). The variational KL-regularization completely
   prevents OOD collapse. RETRO-JEPA-OOD is CLOSED. The next question is: can VJEPA guidance
   be applied during *generation* (streaming) rather than just post-hoc?

2. **SVAMP AUC=0.125 is a labeling mismatch, not a model failure.** SVAMP questions
   ("Tom has 5 apples, gives 2 to Jane, how many left?") are single-step word problems.
   FoVer labeling assumes multi-step CoT chains with intermediate steps to evaluate.
   There is no CoT to label — so all labels are noise. Root cause diagnosed; needs
   a fundamentally different verification approach (estimation verification per arXiv 2509.18565).

3. **RETRO-MANIFEST-FULL-SCOPE has reached 10+ consecutive milestones.** The exclusion
   manifest enforcement patch is documented but not applied to the conductor script in every
   milestone. This costs ~276 min/milestone in wasted doomed-rerun time. Must be closed
   in .69 — Exp 892 is the final attempt before escalating to explicit human intervention.

---

## Architecture Diagram (after .68)

```
LLM Response
    │
    ├─► Tier 0a: CarnotThinkProbe (ThinkPRM, generative CoT verify)
    ├─► Tier 0b: SpilledEnergyDetector (logit discrepancy, AUC=0.97)
    ├─► Tier 0c: NUP Probe v4 (bigram contrastive energy, AUC=1.0)
    ├─► Tier 0d: HallucinationBasinDetector (latent basin depth)
    ├─► Tier 0e: HalluField (token-path ensemble variance, AUC=0.97)
    ├─► Tier 0g: StreamingCoT (.67: PHaS trajectory, advisory)
    ├─► Tier 0h: SpectralAttentionProbe (.68: Laplacian eigenvalue, advisory)
    ├─► Tier 0i: DRIFTProbe [NEW .69] (multi-layer linear probe, advisory)
    ├─► Tier 1: SinkProbe (attention sink concentration)
    ├─► Tier 2: VJEPA v2 (.68: deployed, ood_auc=0.9211)
    │       └─► [.69 target] Tier 2-streaming: VJEPA violation probability fed back
    │           to logit-mask generation guidance in real-time
    ├─► Tier 2.5: SymCodeVerifier (executable arithmetic, AUC=0.804 live)
    ├─► Tier 2.6: HermesVerifierAdapter (step-boundary feedback, candidate)
    ├─► Tier 2.7: CausalReasoningVerifier (causal entailment, recall=0.36)
    ├─► Tier 2.8: DraftConditionedVerifier [NEW .69] (arXiv 2603.03305, draft-conditioned)
    └─► Tier 3: Ising VerifyRepairPipeline (constraint satisfaction)

Self-Learning Loop (FR-11):
    Tier 3 violations → Tier 1 online weight update (Lagrange adaptive, .66) ✓
    Tier 1 memory → Tier 2 JEPA training data (.67 loop closed) ✓
    VJEPA predictions → Constraint addition (.68 closed: Exp 888) ✓
    [.69 target] Tier 4 → Adaptive KAN spline restructuring (FR-11 Tier 4, Exp 898)

SVAMP Estimation Path [NEW .69]:
    Single-step word problem → EstimationVerifier (order-of-magnitude bounds)
    → VJEPA v3 trained on SVAMP-format synthetic pairs
    → Replace noisy FoVer labels with estimation-range labels
```

---

## Milestone .69 Phases

### Phase 0: Governance (Exp 892)

**Exp 892: Pre-flight v18** — PIMI retirement check + manifest full-scope audit + .68 RETRO audit.

Open RETROs entering .69 (4 open after .68):
- RETRO-MANIFEST-FULL-SCOPE: manifest patch documented but never applied to conductor (10+ milestones)
- RETRO-SVAMP-ZERO-AUC: SVAMP AUC=0.125 (root cause: labeling mismatch, not model failure)
- RETRO-XILINX-TOOLS-UNAVAILABLE: KV260 synthesis blocked (Vivado not installed, iCE40 fallback)
- RETRO-INERTIA-SWEEPS-TARGET-MISSED: PIMI parallel updates achieved but <5x, or retired in .68

Key action: if PIMI was retired in Exp 889 (retire_if_same_verdict triggered), mark
RETRO-INERTIA-SWEEPS-TARGET-MISSED as RETIRED (not closed, but no further pursuit).
If Exp 890 (GGUF CLI) was retired, mark RETRO-SOTA-MODEL-DOWNLOAD as CLOSED-BY-RETIREMENT.

RETRO-MANIFEST-FULL-SCOPE: this is now the 11th consecutive milestone it has been open.
Exp 892 must write a one-line test to verify ExclusionManifestEnforcer is invoked from
the conductor path. If enforcement cannot be wired in the conductor without modifying
scripts/research_conductor.py, escalate to ops/known-issues.md with a "human intervention
required" flag.

### Phase 1: SVAMP Root Cause + Estimation Verification (Exp 893 + 896)

**Strategy:** SVAMP AUC=0.125 is a labeling problem, not a model problem. Single-step
word problems have no intermediate CoT steps to label. The fix is to use estimation
verification (arXiv 2509.18565): for a word problem "Tom has 5 apples, gives 2 to Jane,
how many left?", the correct answer is in [0, 10] — simple arithmetic bound check. This
doesn't require multi-step CoT labeling at all.

**Exp 893: SVAMP Root Cause Deep Diagnosis** (CPU)
Confirm the root cause hypothesis with evidence:
- Load 20 SVAMP questions from SVAMPClean (arXiv 2509.18565 cleaned dataset)
- Generate 20 responses from Qwen3.5-0.8B
- Run FoVer labeling pipeline — measure label_noise_rate (fraction of steps with
  ambiguous/impossible labels due to single-step format)
- Measure: mean CoT depth (expected: 1-2 steps vs GSM8K's 5-8 steps)
- Run VJEPA v2 on same 20 — confirm ood_auc near 0.5 (random) for single-step
- Report: labeling_mismatch_confirmed (bool), mean_cot_depth, svamp_auc, gsm8k_auc_for_comparison
- This is diagnostic — no model changes, just evidence gathering
- Gate for Exp 896: labeling_mismatch_confirmed=True required

**Exp 896: SVAMP Estimation Verifier + VJEPA v3 Retrain** (CPU, GATED on Exp 893)
Build a purpose-built verifier for word-problem arithmetic:
- Implement python/carnot/verify/estimation_verifier.py:
  EstimationVerifier extracts: (a) numbers mentioned in question, (b) operation type
  (add/sub/mult/div from question words), (c) plausible answer range
  Returns: {"in_range": bool, "expected_range": [min, max], "actual_answer": float}
- Synthetic labeling for VJEPA v3: run Qwen3.5-0.8B on 100 SVAMP questions,
  label steps by estimation plausibility (not CoT completeness — EstimationVerifier)
- Retrain VariationalJEPAPredictor on SVAMP-specific labeled pairs
- Target: svamp_auc > 0.60 (vs baseline 0.125)
- This closes RETRO-SVAMP-ZERO-AUC if svamp_auc > 0.60

prior_failures:
  - experiment_id: exp883
    verdict: svamp_auc=0.125
    addressed_by: "Root cause confirmed by Exp 893: FoVer labeling inapplicable to
                   single-step word problems. VJEPA v3 uses EstimationVerifier labels
                   instead — fundamentally different labeling strategy."
  - experiment_id: exp872
    verdict: svamp_auc=0.125
    addressed_by: "Same root cause; same fix."
    retire_if_same_verdict: true

### Phase 2: VJEPA Live Streaming + Code Scale-Up (Exp 894 + 895)

**Strategy:** VJEPA ood_auc=0.9211 as a post-hoc filter is proven. The next step is
to move it earlier in the pipeline: use VJEPA violation_probability to guide token
selection during generation (logit masking). This is "streaming" because VJEPA runs
on partial CoT prefixes and feeds back before generation completes.

This is architecturally different from post-hoc filtering: it prevents the model from
going down wrong CoT paths rather than catching them afterward. Connected to arXiv
2502.03685 (Controlled LLM Decoding via Discrete Autoregressive Biasing) and
arXiv 2603.03305 (Draft-Conditioned Constrained Decoding).

**Exp 894: VJEPA Live Streaming Filter** (GPU, CARNOT_FORCE_LIVE=1)
Wire VJEPA as a generation-time logit gate:
- Hook into Gemma4-E4B-it generation via transformers LogitsProcessor API
- At each generation step: encode current CoT prefix via VJEPA encoder
- Compute p(violation) = VJEPA.violation_probability(current_prefix)
- If p(violation) > threshold (0.75): apply soft penalty to logits for tokens that
  increase KL divergence (VJEPA-guided masking)
- Run 25 GSM8K questions with streaming filter active vs baseline
- Measure: streaming_filter_applied (bool), gsm8k_accuracy_baseline,
  gsm8k_accuracy_streaming, signed_improvement, avg_tokens_saved_pct
- Target: signed_improvement > 0 AND tokens_saved_pct > 5%
- CARNOT_FORCE_LIVE=1 required

prior_failures:
  - experiment_id: exp888
    verdict: fr11_tier3_relay (not streaming)
    addressed_by: "Exp 888 wired VJEPA to constraint ADDITION (post-hoc). This experiment
                   wires VJEPA to logit MASKING during generation (pre-hoc). Different
                   integration point: generation loop vs verification loop."

**Exp 895: Code Repair 50-question Scale-Up** (GPU, GATED on Exp 881 signed_improvement > 0)
Scale the Exp 881 code repair to 50 HumanEval problems:
- Use same Gemma4-E4B-it + transformers loader pipeline as Exp 881
- Run 50 HumanEval problems (vs 25 in Exp 881)
- Apply full CodeExtractor + VerifyRepairPipeline + ConstrainedDecodingPreFilter (Exp 886)
- Measure: pass_at_1_baseline, pass_at_1_carnot, signed_improvement, repair_success_rate
- CARNOT_FORCE_LIVE=1 required
- This is the first full-scale live code verification result

### Phase 3: New Capabilities (Exp 897 + 899 + 900)

**Exp 897: Constraint Memory Forgetting — FOREVER-Inspired Lagrange Weight Decay** (CPU)
The current Lagrange weight update accumulates constraints indefinitely. In a long session,
early constraints from the first 10 questions dominate the energy landscape even when
they are no longer relevant. Inspired by arXiv 2601.03938 (FOREVER forgetting curve replay):
- Implement forgetting curve for constraint memory in python/carnot/pipeline/lagrange_updater.py:
  Add per-constraint age tracking and exponential decay: w_t = w_0 * exp(-lambda * age_t)
  where lambda (forgetting rate) is tunable per constraint type
- Optionally: replay high-violation constraints from memory before forgetting them
  (FOREVER-style: replay items that are about to be forgotten, at high violation probability)
- Run 10-session relay: 20 questions/session, compare memory-with-forgetting vs
  memory-without-forgetting on constraint precision
- Measure: constraint_precision_no_forget, constraint_precision_with_forget,
  memory_size_no_forget, memory_size_with_forget, forgetting_rate_best_lambda

**Exp 899: DRIFT Hidden-State HalluProbe — Tier 0i Multi-layer Linear Probe** (CPU)
Based on arXiv 2601.14210 (DRIFT: multi-layer representational inconsistency):
- For each response, extract hidden states from layers {4, 8, 12, 16} of Qwen3.5-0.8B
  (or Gemma4-E4B-it if GPU available)
- Compute representation drift: cosine inconsistency between adjacent-layer embeddings
  for the same token position
- Train linear probe: drift_signature → is_hallucination (on 50 FoVer labeled pairs)
- Wire as Tier 0i advisory in VerificationCertificate (flag is_representationally_drifted)
- Target AUC > 0.65; CPU-only (use Qwen3.5-0.8B activation extraction)
- Advisory only — does not short-circuit cascade

**Exp 900: Draft-Conditioned Verifier — Tier 2.8** (CPU)
Based on arXiv 2603.03305 (Draft-Conditioned Constrained Decoding):
- Implement DraftConditionedVerifier:
  Generate a cheap draft response (Qwen3.5-0.8B, 3 tokens/step lookahead)
  Condition the full-generation constraint set on the draft's structure
  When VJEPA predicts the draft path violates constraints: block that generation branch
- Implement as Tier 2.8 between Tier 2.7 and Tier 3
- Run 20 GSM8K questions with draft-conditioned verification
- Measure: draft_mismatch_rate, constraint_violation_pre_correction,
  constraint_violation_post_correction, signed_improvement
- Target: constraint_violation_post_correction < constraint_violation_pre_correction

### Phase 4: Hardware (Exp 901)

**Exp 901: PIMI Sparse Adjacency v4 — Copy-Node Sparsification** (CPU + iCE40)

This is the FINAL attempt at the PIMI sweeps improvement target.

**Context:** Exp 889 (parallel updates) was the last attempt under the parallel-update
hypothesis. If Exp 889 achieved sweeps_reduction in [3, 5), the hypothesis was partially
correct but insufficient. The remaining headroom is in the connectivity structure.
arXiv 2503.01177 proposes "copy-node sparsification" — replacing dense all-to-all
couplings with a sparse star-topology via an intermediate copy-node layer. This reduces
the effective coupling count from O(N^2) to O(N*k) while preserving solution quality
for k-sparse problems.

If Exp 889 produced honest_verdict="pimi_retired" (sweeps_reduction < 3): this experiment
is ALSO retired. The prior_failures check below handles this.

prior_failures:
  - experiment_id: exp889
    verdict: pimi_improved_below_5x (expected)
    addressed_by: "Sparse adjacency via copy-node layers (arXiv 2503.01177) reduces
                   effective coupling count O(N^2) -> O(N*k); hypothesis: dense
                   coupling creates spurious energy minima that increase sweep count
                   independent of update strategy."
    retire_if_same_verdict: true
  - experiment_id: exp876
    verdict: sweeps_improved_below_5x
    addressed_by: "Combined sparse adjacency (new) + parallel updates (from Exp 889)."
  - experiment_id: exp860
    verdict: sweeps_improved_below_5x
    addressed_by: "Same root cause chain."

If Exp 889 verdict was "pimi_retired": Exp 901 must also retire without running.
Write artifact with honest_verdict="pimi_retired_upstream" and add iCE40 PIMI to
ops/exclusion_manifest.yaml (if not already there from Exp 889).

### Phase 5: FR-11 Tier 4 Seed (Exp 898)

**Exp 898: FR-11 Tier 4 Seed — Adaptive KAN Spline Restructuring** (CPU)

FR-11 Tier 4 is the last unstarted tier of the self-learning loop: the KAN model
(python/carnot/models/kan.py) should restructure its spline grids in response to
accumulated violation statistics — adding finer spline resolution in regions of
high violation density, and coarsening in regions with no violations.

This is the seed experiment: does the KAN actually benefit from adaptive restructuring?

- Run KAN in standard mode on 100 FoVer pairs. Record per-spline activation histograms.
- Identify high-density regions (top 10% of activations) and low-density regions
  (bottom 10% of activations).
- Implement KANAdaptiveStructure:
  high_density_region → double the spline knot count (refine)
  low_density_region → halve the spline knot count (coarsen)
  neutral_region → no change
- Retrain KAN after restructuring on same FoVer pairs. Measure: energy_loss_before,
  energy_loss_after, knot_count_change_pct, tier1_alignment_delta.
- Target: energy_loss_after < energy_loss_before (adaptive restructuring helps).
- This is a seed — success means Tier 4 is viable; failure means Tier 4 approach changes.

### Phase 6: Publication + Retrospective (Exp 902 + 903)

**Exp 902: HuggingFace Publish v3** (CPU)
Update HuggingFace model cards (huggingface.co/Carnot-EBM) to reflect .68 results:
- VJEPA v2 model weights and card: ood_auc=0.9211, training corpus 207 pairs
- Updated README with Three-Tier pipeline diagram (reflecting .68 architecture)
- Update model card: NUP Probe v4 (AUC=1.0), SpectralAttentionProbe (Tier 0h)
- Ensure all published weights have IPFS mirror (decentralization rule 3)
- Write publish_confirmed=True and ipfs_mirror_cid to artifact

**Exp 903: Milestone .69 Operational Retrospective** (CPU)
Compute wall time, per-experiment average, slowest-5, criteria met, retros closed.
Evaluate: RETRO-MANIFEST-FULL-SCOPE, RETRO-SVAMP-ZERO-AUC, RETRO-XILINX-TOOLS-UNAVAILABLE,
RETRO-INERTIA-SWEEPS-TARGET-MISSED. Write artifact and update ops/status.md,
ops/changelog.md.

---

## Dependency Graph

```
Exp 892 (preflight)
    │
    ├──► Exp 893 (SVAMP root cause diagnosis)
    │       │
    │       └──► Exp 896 (SVAMP estimation verifier + VJEPA v3, GATED on 893)
    │
    ├──► Exp 894 (VJEPA live streaming, GPU)
    │
    ├──► Exp 895 (code repair 50q scale-up, GPU, GATED on Exp 881 result from .68)
    │
    ├──► Exp 897 (Lagrange forgetting curve, CPU-only, parallel with all)
    │
    ├──► Exp 898 (FR-11 Tier 4 KAN seed, CPU-only, parallel with all)
    │
    ├──► Exp 899 (DRIFT probe Tier 0i, CPU-only, parallel with all)
    │
    ├──► Exp 900 (draft-conditioned verifier Tier 2.8, CPU-only, parallel with all)
    │
    ├──► Exp 901 (PIMI sparse adjacency v4 — FINAL; ABORTS if Exp 889 pimi_retired)
    │
    └──► Exp 902 (HF publish v3, CPU, after 894/895/896 results available)
         │
         └──► Exp 903 (retrospective, reads all result JSONs)
```

---

## Success Criteria

| # | Criterion | Target | Closes RETRO? |
|---|-----------|--------|---------------|
| 1 | manifest_enforcement_verified | Exp 892 enforcement_wired=True OR escalation_documented | RETRO-MANIFEST-FULL-SCOPE |
| 2 | svamp_root_cause_confirmed | Exp 893 labeling_mismatch_confirmed=True | — |
| 3 | vjepa_streaming_positive | Exp 894 signed_improvement > 0 | — |
| 4 | code_repair_50q_positive | Exp 895 signed_improvement > 0 | — |
| 5 | svamp_auc_above_threshold | Exp 896 svamp_auc > 0.60 | RETRO-SVAMP-ZERO-AUC |
| 6 | lagrange_forgetting_improves | Exp 897 constraint_precision_with_forget > constraint_precision_no_forget | — |
| 7 | kan_tier4_viable | Exp 898 energy_loss_after < energy_loss_before | — |
| 8 | drift_probe_viable | Exp 899 probe_auc > 0.65 | — |
| 9 | draft_conditioned_verifier_viable | Exp 900 signed_improvement > 0 | — |
| 10 | pimi_resolved | Exp 901 sweeps_reduction >= 5.0 OR honest_verdict contains "retired" | RETRO-INERTIA-SWEEPS-TARGET-MISSED |
| 11 | hf_publish_complete | Exp 902 publish_confirmed=True AND ipfs_mirror_cid != null | — |

Target: 7+ criteria met, 2+ retros closed.

---

## Hardware Requirements

- **Exp 894:** GPU required — Gemma4-E4B-it streaming; CARNOT_FORCE_LIVE=1
- **Exp 895:** GPU required — Gemma4-E4B-it 50q code repair; CARNOT_FORCE_LIVE=1
- **All others:** CPU only (JAX_PLATFORMS=cpu)
- **Exp 901:** CPU for simulation + OSS-CAD-Suite for iCE40 synthesis (if not pimi_retired)

---

## Failed-Experiment Rerun Compliance

All experiments referencing prior failures comply with CLAUDE.md's Failed-Experiment
Rerun Discipline:

- **Exp 896** (SVAMP v3): prior_failures [exp883, exp872]. addressed_by: "Estimation
  verifier labels instead of FoVer CoT labels — root cause confirmed by Exp 893."
  retire_if_same_verdict=true.
- **Exp 901** (PIMI v4): prior_failures [exp889, exp876, exp860]. addressed_by: "Sparse
  adjacency copy-node sparsification (arXiv 2503.01177) — different structural hypothesis
  from parallel-update hypothesis (Exp 889)." retire_if_same_verdict=true.
  ABORTS if Exp 889 produced pimi_retired.

## RETRO Status Entering .69

| RETRO ID | Status | Age (milestones) | .69 action |
|----------|--------|-----------------|------------|
| RETRO-MANIFEST-FULL-SCOPE | OPEN | 11 | Exp 892: final enforcement attempt; escalate to human if blocked |
| RETRO-SVAMP-ZERO-AUC | OPEN | 3 | Exp 893+896: estimation verifier approach |
| RETRO-XILINX-TOOLS-UNAVAILABLE | OPEN | 8+ | No action (Vivado install requires human action) |
| RETRO-INERTIA-SWEEPS-TARGET-MISSED | OPEN or RETIRED | 3 | Exp 901: sparse adjacency final attempt |
| RETRO-JEPA-OOD | CLOSED (.68) | — | — |
| RETRO-HALLUSAE-AUC-BELOW-THRESHOLD | CLOSED (.68) | — | — |
| RETRO-SOTA-MODEL-DOWNLOAD | CLOSED (.68) | — | — |
