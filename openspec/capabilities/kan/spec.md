# KAN Distillation Capability Specification

**Capability:** kan
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-KAN-001, FR-KAN-002, FR-KAN-003, FR-KAN-004

## Overview

Defines how Carnot distills a large teacher safety model (gpt-oss-safeguard-20b)
into a compact KAN (Kolmogorov-Arnold Network) classifier for real-time prompt
injection detection.  KAN spline activations are interpretable: each control
point maps directly to a named injection pattern feature, so a human auditor
can read the spline to understand why a prompt was flagged.

The distillation pipeline trains the KAN on teacher-labeled examples, measures
AUROC on a held-out split, and gates deployment on reaching a minimum quality
threshold (REQ-KAN-003/004).

## Requirements

### REQ-KAN-003: Distillation Training Dataset >= 3000 Examples

The KAN distillation training dataset MUST contain at least 3000 labeled
examples with a balanced positive/negative ratio (1500 injection + 1500 benign).

**Rationale:**
    Exp 710 trained on 1091 examples and achieved AUROC=0.8747, just below the
    0.90 Tier 0b deployment gate.  Variance analysis shows the gap is driven by
    dataset size: with fewer than 1500 examples per class the contrastive loss
    surface is noisy and the optimizer settles in a shallow basin.  3000 balanced
    examples provide enough signal for the 16-knot splines to fit the teacher's
    classification boundary.

**Acceptance criteria:**
    - `len(dataset) >= 3000`
    - `count(label="injection") == count(label="benign")` (within ±1)
    - Dataset persisted at `results/kan_distill_v3_dataset.json` with schema field.

### SCENARIO-KAN-003: Dataset balance check

Given a generated dataset,
When the loader reads `results/kan_distill_v3_dataset.json`,
Then `len(examples) >= 3000` AND `abs(n_positive - n_negative) <= 1`.

### REQ-KAN-004: KAN Architecture >= 16 Knots Per Activation

The KAN distillation model used for Tier 0b quality gating MUST use at least
16 spline knots per activation.

**Rationale:**
    8 knots (v2) could not capture the fine decision boundary near the teacher's
    0.90 AUROC threshold.  16 knots double the spline resolution, allowing the
    energy landscape to model sharper transitions between benign and injection
    regions.  The interpretability property is preserved: each knot corresponds
    to a piecewise-linear segment of the activation, readable as a breakpoint
    in the injection-feature sensitivity curve.

    Parameter count with 16 knots, n_hidden=8, n_features=32, degree=3:
        edge_params  = 8 * 32 * (16+3) = 8 * 32 * 19 = 4864
        output_params = 8 * 19          = 152
        total         = 5016

**Acceptance criteria:**
    - `PromptInjectionEnergyCheckerV3._N_KNOTS == 16`
    - `checker.n_params() == 5016` with default constructor arguments.
    - Training for 100 epochs on 3000 examples completes without error.

### SCENARIO-KAN-004: 16-knot architecture sanity check

Given `PromptInjectionEnergyCheckerV3()`,
When we call `checker.n_params()`,
Then the result is 5016 (= 8 * 32 * 19 + 8 * 19).

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-KAN-003 | Proposed | Exp 724 target: 3000-example balanced dataset |
| REQ-KAN-004 | Proposed | Exp 724 target: 16 knots/spline in v3 KAN |
| REQ-KAN-020 | Implemented | Exp 866: LUT analysis complete. N=8 MLP-bound=14400 > 7680 (over budget). ISING_PRIORITY. |
| REQ-MODEL-SOS-001 | Implemented | Exp 1047: SOSKANEnergy confirmed 0 violations on 16000 test points. |
| REQ-KAN-1148 | Proposed | Exp 1148 target: MetaCluster-style centroid compression for SOSKANEnergyV3 with AUROC drop <= 0.02 and >=5x shrink. |
| REQ-KAN-1162 | Implemented | Exp 1162: KANELE-style Q8 LUT blueprint and hardware-complexity artifact generated for the compressed SOSKANEnergyV3 shape. |
| REQ-KAN-1174 | Proposed | Exp 1174 target: BiKA multiply-free complexity analysis for SOSKANEnergyV3, MetaCluster, and AMD XDNA NPU feasibility. |
| REQ-KAN-1199 | Proposed | Exp 1199 target: KANtize-style 8-bit/4-bit SOSKANEnergyV3 spline quantization with endpoint-sensitive precision and safetensors export. |

## REQ-KAN-020: KAEMEnergy FPGA LUT Budget Analyzability

KAEMEnergy MUST be analyzable for FPGA LUT budget using arXiv 2604.03345
per-knot estimates. Target: N=8, within iCE40 HX8K 7680 LUT budget.

**Rationale:**
    The iCE40 HX8K has 7,680 LUTs.  Before committing synthesis effort, a
    conservative LUT estimate is required.  arXiv 2604.03345
    (Hardware-Oriented KAN Inference Complexity) provides the formula:
        LUTs = fan_in * fan_out * n_knots * luts_per_segment
    where luts_per_segment = 8–12 (we use 10 as conservative midpoint).

    Exp 866 result (2026-04-25):
      - KAN N=8 MLP upper bound: 14,400 LUTs (over 7,680 budget)
      - Ising N=8 actual (Exp 859): 134 LUTs
      - Synthesis priority: ISING_PRIORITY
      - Note: actual KAEMEnergy is sparse graph-based (edge_density=0.1),
        so the real LUT cost is ~10% of the MLP estimate (~1,440 LUTs),
        which would fit within budget.  A sparse-aware analysis is needed
        to confirm feasibility definitively.

**Acceptance criteria:**
    - `KANHardwareAnalyzer` in `python/carnot/analysis/kan_hw_analysis.py`
      implements `lut_estimate_layer()`, `total_lut_estimate()`,
      `synthesis_priority()`, and `sensitivity_analysis()`.
    - `total_lut_estimate()` returns dict with keys:
      layer1_luts, layer2_luts, total_luts, ice40_hx8k_budget, within_budget.
    - `synthesis_priority()` correctly returns one of:
      KAN_PRIORITY / ISING_PRIORITY / BOTH_FEASIBLE.

### SCENARIO-KAN-030: LUT estimate for N=8 KAEMEnergy on iCE40 HX8K

Given KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10),
When total_lut_estimate() is called,
Then layer1_luts=12800, layer2_luts=1600, total_luts=14400,
     within_budget=False (14400 > 7680).

## REQ-MODEL-SOS-001: SOSKANEnergy MUST guarantee ψ'(x) >= 0 as a type-level property

SOSKANEnergy MUST guarantee ψ'(x) >= 0 (monotonicity) as a type-level property,
with zero violations possible regardless of parameter values V and c.

**Rationale:**
    KAEMEnergy's post-hoc isotonic projection (enforce_monotonicity) fixes violations
    AFTER they occur during training. This is the wrong framing: violations can
    accumulate within an epoch, the MILP verifier must re-run after each epoch, and
    the projection changes the energy landscape in a non-gradient direction.

    SOSKANEnergy uses the SOS (Sum-of-Squares) parameterization:
        ψ'(x) = ||V^T B(x)||² = B(x)^T (V V^T) B(x) >= 0
    for any unconstrained V. No projection, no constraint, no post-hoc repair.
    The verifier can be run ONCE and the invariant holds forever.

    Exp 1047 confirmed: 0 monotonicity violations on 16,000 random test points
    (1,000 × 16 features) with adversarially large random V matrices.

**Acceptance criteria:**
    - `SOSKANEnergy(n_sos_basis=2, ...)` instantiates without error.
    - `verify_invariants(n_samples=1000)` returns `n_monotone_violations == 0`
      for any V (including V set to large random values post-construction).
    - `forward(x) >= 0` for any x in [-1, 1]^n_features.
    - AUROC >= 0.50 on FoVer corpus (no regression from KAEMEnergy baseline).
    - Training is 5x faster than KAEMEnergy (no projection overhead).

### SCENARIO-MODEL-SOS-001: zero violations on adversarial V

Given SOSKANEnergy with V overwritten to random N(0, 10) values,
When verify_invariants(n_samples=1000) is called,
Then n_monotone_violations == 0 and invariants_hold == True.

## REQ-KAN-1148: SOSKANEnergyV3 centroid codebook compression

SOSKANEnergyV3 compression MUST replace repeated learned coefficient vectors
with a K=32 centroid codebook plus per-vector centroid indices, then reconstruct
a numerically usable SOSKANEnergyV3 instance from that compressed payload.

**Rationale:**
    Exp 1128 fixed the SOSKANEnergyV3 training/inference normalization path and
    reported individual AUROC=0.9902 on the 500-example FoVer benchmark.  That
    verifier is now accurate enough to matter, but its neural-Gram coefficient
    head is larger than ideal for cheap-tier routing and NPU/FPGA deployment.
    MetaCluster-style centroid codebooks test whether the repeated KAN
    coefficient vectors can be stored compactly while preserving the energy
    ordering that drives AUROC.

**Acceptance criteria:**
    - `compress_sos_kan_v3(..., n_centroids=32)` stores float centroid arrays
      and integer index arrays instead of dense coefficient vectors.
    - `reconstruct_sos_kan_v3()` returns an SOSKANEnergyV3 with the same
      architecture fields and finite energies.
    - The exp1148 artifact records `auroc_original=0.9902`,
      `auroc_compressed`, `auroc_drop`, `energy_correlation`,
      `size_original_bytes`, `size_compressed_bytes`, and
      `size_reduction_factor`.
    - `auroc_drop_within_02` is true iff `auroc_original - auroc_compressed <= 0.02`.
    - The success verdict requires both `auroc_drop <= 0.02` and
      `size_reduction_factor >= 5.0`.

### SCENARIO-KAN-1148: reconstruct SOSKANEnergyV3 from centroid codebook

Given a trained SOSKANEnergyV3 and its learned coefficient vectors,
When the vectors are clustered into 32 centroids and reconstructed by expanding
indices back to centroid vectors,
Then the reconstructed model has the original parameter shapes, finite energies
on FoVer features, and an exp1148 artifact with the required verdict fields.

## REQ-KAN-1162: SOSKANEnergyV3 KANELE FPGA LUT blueprint

Experiment 1162 MUST generate a KANELE-style FPGA blueprint for the compressed
SOSKANEnergyV3 energy function shape from Exp 1148 without requiring RTL
synthesis.  The blueprint MUST identify the SOS-KAN input count, spline-basis
count, knot-grid size, Q8 LUT storage, hardware-oriented complexity metrics
(`RM`, `BOP`, `NABS`), KV260 latency estimate, CPU-baseline comparison, and the
compressed AUROC inherited from Exp 1148.

**Rationale:**
    Exp 1148 showed that the fixed Exp 1128 SOSKANEnergyV3 shape can be reduced
    by a K=32 centroid codebook while keeping AUROC within 0.02 of the original
    model.  KANELE-style FPGA work maps KAN univariate spline basis functions to
    LUTs.  Before RTL exists, Carnot needs a deterministic table specification
    and platform-independent complexity report that can be reviewed without
    Vivado.

**Acceptance criteria:**
    - The runner derives `sos_kan_n_inputs`, `sos_kan_k_splines`, and
      `sos_kan_n_knots` from SOSKANEnergyV3/Exp 1148 structure rather than
      hard-coding an unrelated architecture.
    - The Q8 table specification samples every hat basis function at
      `n_lut_points=256` uniformly over `[-1, 1]` and reports
      `lut_storage_bytes = n_inputs * k_splines * n_lut_points`.
    - The artifact reports `rm_per_inference`, `bop_per_inference`,
      `nabs_per_inference`, `estimated_fpga_latency_us`,
      `cpu_baseline_latency_ms=289.0`, `estimated_speedup_factor`,
      `blueprint_written`, `blueprint_path`, `auroc_compressed`,
      `kanele_fpga_blueprint_generated`, and an approved honest verdict.
    - `hardware/kv260/sos_kan_lut_blueprint.md` documents the LUT index formula,
      Q8 interpolation datapath, accumulation schedule, and no-Vivado status.

### SCENARIO-KAN-1162: write compressed SOS-KAN Q8 LUT blueprint

Given the Exp 1148 compressed SOSKANEnergyV3 result with K=32 centroids,
When `scripts/experiment_1162_kanele_sos_kan_fpga_blueprint.py` runs,
Then `results/experiment_1162_kanele_sos_kan_fpga_blueprint.json` is written
with the required schema fields and
`hardware/kv260/sos_kan_lut_blueprint.md` contains the deterministic Q8 LUT
blueprint for the SOS-KAN basis structure.

## REQ-KAN-1174: BiKA multiply-free SOS-KAN hardware analysis

Experiment 1174 MUST analyze whether the Exp 1148/1162 SOSKANEnergyV3 shape can
be mapped to a BiKA-style multiply-free datapath by replacing floating-point
multiplications with precomputed-log2 bit-shift approximations.  The analysis
MUST report platform-independent `RM`, `BOP`, and `NABS` metrics for the
standard SOSKANEnergyV3, the MetaCluster-compressed SOS-KAN, and the
8-bit BiKA approximation, then classify AMD XDNA NPU feasibility.

**Rationale:**
    Exp 1162 proved that the compressed SOS-KAN spline structure can be described
    as Q8 LUT tables for KV260.  BiKA is the arithmetic layer below that
    blueprint: replacing learned-coefficient multiplies with shifts/adds can
    remove the float32 multiplier requirement for AMD XDNA NPUs and reduce FPGA
    resource pressure before RTL synthesis.

**Acceptance criteria:**
    - `BiKAComplexityAnalyzer.analyze_standard_kan(model)` returns a
      `HardwareMetrics` object with integer `RM`, `BOP`, `NABS`, and
      `estimated_lut_count` values derived from SOSKANEnergyV3 architecture.
    - `BiKAComplexityAnalyzer.analyze_bika_kan(model, precision_bits=8)` returns
      zero real multiplications and models BOP as `standard_RM * 16` for the
      8-bit shift-plus-comparison approximation.
    - `BiKAComplexityAnalyzer.compare(standard_metrics, bika_metrics)` reports a
      resource-reduction percentage in the 27.73% to 51.54% BiKA paper band and
      an NPU feasibility verdict in the approved vocabulary.
    - `scripts/experiment_1174_bika_hardware_analysis.py` writes
      `results/experiment_1174_bika_hardware_analysis.json` with
      `standard_kan_rm`, `standard_kan_bop`, `compressed_kan_rm`,
      `compressed_kan_bop`, `bika_kan_nabs`, `bika_resource_reduction_pct`,
      `npu_feasibility_verdict`, `estimated_npu_inference_us`,
      `bika_hardware_analysis_complete`, and an approved honest verdict.

### SCENARIO-KAN-1174: write BiKA hardware analysis artifact

Given the Exp 1148 MetaCluster compression artifact and Exp 1162 KANELE LUT
blueprint,
When `scripts/experiment_1174_bika_hardware_analysis.py` runs,
Then `results/experiment_1174_bika_hardware_analysis.json` is written with the
required schema fields, the SOS-KAN architecture values, multiply-free BiKA
metrics, and an honest AMD XDNA NPU feasibility verdict.

## REQ-KAN-1199: KANtize-style SOSKANEnergyV3 spline quantization

Experiment 1199 MUST quantize SOSKANEnergyV3 spline-control parameters at 8-bit
and 4-bit precision while assigning endpoint spline control points twice the
interior bit width, then evaluate AUROC, memory size, latency, and deployment
readiness on the FoVer holdout.

**Rationale:**
    Exp 1128 established a full-precision SOSKANEnergyV3 AUROC reference of
    0.9902 on the 500-example FoVer benchmark.  KANtize-style low-bit spline
    quantization tests whether the same verifier can be represented cheaply
    enough for consumer edge hardware.  Endpoint spline values are quantized
    with higher precision because endpoint perturbations are more sensitive
    than interior perturbations in spline functions.

**Acceptance criteria:**
    - Interior SOSKANEnergyV3 spline rows SHALL be rounded to the nearest
      `1/(2^bits - 1)` interval for `bits in {8, 4}`.
    - First and last spline-control rows SHALL be rounded separately with
      `2 * bits` precision.
    - Non-spline head parameters SHALL remain exact in the quantized model.
    - `scripts/experiment_1199_kantize_soskan_4bit_quantization.py` SHALL write
      `results/experiment_1199_kantize_soskan_4bit_quantization.json` with
      full-precision, 8-bit, and 4-bit AUROC and size fields, 4-bit latency,
      safetensors checkpoint path, threshold booleans, and an honest verdict.
    - The 4-bit safetensors export SHALL contain a quantized SOSKANEnergyV3
      checkpoint with architecture and quantization metadata.

### SCENARIO-KAN-1199: evaluate and export quantized SOSKANEnergyV3

Given the Exp 1128 SOSKANEnergyV3 FoVer setup,
When `scripts/experiment_1199_kantize_soskan_4bit_quantization.py` runs,
Then it evaluates 32-bit, 8-bit, and 4-bit AUROC, measures packed model sizes
and 4-bit per-sample latency, exports the 4-bit quantized checkpoint as
safetensors, and reports whether `soskan_4bit_auroc >= 0.97`.
