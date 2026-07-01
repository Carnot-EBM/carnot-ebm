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

## REQ-KAN-VERIFY-001: KAEM/KAN PWA And MILP Property Verification

The repository shall provide CPU-only KAN formal-verification helpers for small
KAEM/KAN energy layers. The helpers shall:

- split spline control points into piecewise-affine segments;
- verify monotonicity by direct knot inspection;
- verify output range and boundary-condition properties;
- perform a MILP or equivalent linear counterexample search for monotonicity;
  and
- return structured result dictionaries with a `verified` boolean and violation
  detail fields.

The verification helpers shall make no hardware correctness claim. They are
software property checks used to audit KAN energy functions before deployment
or repair.

### SCENARIO-KAN-VERIFY-001: Small KAN Properties Are Audited

Given a small KAN layer with flat, monotone, and intentionally non-monotone
control points,
When the PWA, monotonicity, range, boundary, and MILP helpers run,
Then flat and monotone layers are verified, non-monotone or inverted layers
report violations, and every result exposes deterministic keys suitable for a
terminal experiment artifact.

## REQ-KAN-2871: Tiny KAN PWA/MILP Verifier Prototype With Honest Solver Boundary

The KAN verification tier SHALL provide a tiny CPU-only prototype that replaces
one bounded KAN-style univariate unit with an explicit piecewise-affine
abstraction, records local and global abstraction error bounds, and checks one
output property through either a local MILP solver or an exact finite
enumeration fallback when no MILP solver is available.

The prototype MUST write `results/experiment_2871_kan_pwa_milp_tiny_verifier_v1.json`
with explicit solver/proof boundaries. It MUST NOT claim a general KAN verifier,
trained-network soundness, or MILP readiness when the run used only the exact
fallback.

### SCENARIO-KAN-2871: Tiny PWA Property Artifact

Given the deterministic fixture `phi(x) = x^2` on `x in [-1, 1]` with a bounded
property domain `x in [-0.5, 0.5]`,
When the verifier builds knot-aligned PWA chord envelopes and checks
`phi(x) <= 0.25`,
Then the artifact reports the exact local/global error bounds, records whether
a MILP solver or exact fallback was used, verifies the property only when the
certified upper bound is within the threshold, and includes all required Exp
2871 schema fields.

## REQ-KAN-2876: KAN PWA/MILP Corrigendum With Non-Tautological Bounds

The KAN verification tier SHALL provide a narrow Exp 2876 corrigendum for the
Exp 2871 tiny PWA/MILP prototype. The corrigendum MUST compute the local
abstraction error bound and the global output error bound through distinct
procedures, and the result artifact MUST record whether the two bounds are
distinct by construction.

The corrigendum MUST attempt a real local mixed integer linear backend before
using exact PWA vertex enumeration. If no supported solver dependency is
available, the artifact MUST report `blocked_solver_dependency` and may include
exact enumeration only as fallback evidence. The artifact MUST NOT claim MILP
readiness when solver dependencies are absent.

### SCENARIO-KAN-2876: Corrigendum Clears Exp 2871 Tautology Flag

Given a deterministic two-unit KAN-style PWA fixture with positive output
weights,
When the verifier computes per-segment local residual bounds, propagates a
global output error bound through the weighted output graph, and maximizes the
PWA upper envelope through a local solver when available,
Then `local_error_bound` and `global_error_bound` are not mechanically equal,
`bounds_distinct_by_construction` is true, solver availability and status are
reported, exact enumeration is marked fallback-only, and
`results/experiment_2876_kan_pwa_milp_corrigendum_v2.json` includes every
required schema field.

## REQ-KAN-2893: Tiny KAN PWA/MILP Hardware-Oriented Complexity Accounting

The KAN verification tier SHALL provide a no-hardware-claim complexity
accounting pass for the clean Exp 2876 tiny KAN PWA/MILP fixture. The accounting
MUST load the Exp 2876 artifact, derive the two-unit PWA structure from the
artifact or deterministic fixture, and report platform-independent operation and
structural counts inspired by arXiv:2604.03345: real multiplications, bit
operations, additions and bit-shifts, memory table entries, PWA regions, and
MILP branch/constraint counts.

The artifact MUST be written to
`results/experiment_2893_kan_hardware_complexity_accounting_v1.json`, MUST
compare against local KANELE/QuantKAN/KAEM accounting helper conventions where
applicable, and MUST explicitly state that no FPGA, analog KAN, board, synthesis,
or hardware execution claim is made.

### SCENARIO-KAN-2893: Tiny PWA Complexity Artifact Is Deterministic

Given the completed Exp 2876 two-unit KAN PWA/MILP corrigendum artifact,
When the hardware-oriented accounting helper runs,
Then the Exp 2893 artifact records deterministic RM/BOP/NABS, memory-table,
PWA-region, MILP constraint, and branch counts; includes source artifacts,
field-principle notes, local and global error bounds inherited from Exp 2876;
sets `hardware_execution_claim_made=false` and `analog_kan_claim_made=false`;
and contains every required schema field.

## REQ-KAN-2904: KV260-Anchored KAN Hardware Complexity Accounting V2

The KAN verification tier SHALL provide an Exp 2904 accounting pass that
aggregates completed upstream artifacts instead of running new synthesis or
board commands. The pass MUST read the tiny KAN node count from the Exp 2893 KAN
PWA/MILP accounting artifact and the KV260 LUT, BRAM, and DSP utilization counts
from the Exp 2898 bitstream/report evidence. It MUST write
`results/experiment_2904_kan_hardware_complexity_accounting_v2.json` with:
`honest_verdict`, `inference_substrate`, `kan_node_count`, `kv260_lut_used`,
`kv260_bram_used`, `kv260_dsp_used`, `scaling_estimate_to_next_size`,
`cited_upstream_artifacts`, and `duration_s`.

The artifact MUST set
`inference_substrate="aggregation_from_upstream_artifacts"` and MUST explicitly
state that the scaling estimate is a conservative proxy derived from upstream
KV260 bitstream utilization, not a KAN synthesis, timing-closure, or new board
execution claim.

### SCENARIO-KAN-2904: KV260 Utilization Refines KAN Scaling Estimate

Given the completed Exp 2893 KAN accounting artifact and the Exp 2898 KV260
bitstream/utilization evidence,
When the Exp 2904 aggregation helper runs,
Then the artifact records the Exp 2893 KAN node count, the Exp 2898 KV260 LUT,
BRAM, and DSP counts, a deterministic next-size scaling estimate, the cited
upstream artifact paths and checksums, and every required schema field.

## REQ-KAN-3131: KAN PWA/MILP Verifier Abstraction Audit V1

The KAN verification tier SHALL provide a bounded Exp 3131 audit that inspects
whether local KAN/KAEM verifier code exists, builds a tiny piecewise-affine
abstraction from the existing CPU-only fixture when it does, and records local
and global error-bound accounting plus a MILP-compatible property check.

The audit MUST write
`results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json` with
the following schema fields: `kan_pwa_milp_audit_v1_ready`,
`kan_code_present`, `abstraction_count`, `local_error_bound_summary`,
`global_error_bound_summary`, `milp_property_check_count`,
`milp_property_pass_count`, `implementation_blockers`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

If local KAN/PWA verifier code is missing, the artifact MUST fail closed with a
deterministic implementation-boundary contract that names the exact missing
modules, schemas, and tests required before implementation. If local code is
present, the audit MUST keep claims bounded to abstraction accounting and MUST
NOT claim deployed verifier improvement, trained-network soundness, hardware
execution, model-weight updates, or live LLM inference.

### SCENARIO-KAN-3131: Existing Tiny KAN PWA/MILP Fixture Produces Bounded Audit

Given the existing Exp 2876 two-unit KAN-style PWA/MILP fixture and the local
OpenSpec KAN capability,
When the Exp 3131 audit runs on CPU,
Then the artifact reports KAN code presence, abstraction count, explicit
per-unit local error bounds, propagated global output bounds, MILP-compatible
property-check pass/fail counts, source artifact provenance, no live inference
substrate, no implementation blockers, and an honest terminal verdict with no
deployed verifier improvement claim.

## REQ-KAN-3145: KAN Proof-Carrying Monitor Boundary V2

The KAN verification tier SHALL provide a bounded Exp 3145 proof-carrying
monitor boundary that attaches replayable KAN PWA/MILP proof records to a tiny
subset of existing fragment-time monitor fixtures. The boundary MUST reuse the
local Exp 3131 KAN abstraction evidence when code is present and MUST keep
`deployed_verifier_claim=false`.

The artifact MUST be written to
`results/experiment_3145_kan_proof_carrying_monitor_boundary_v2.json` with
the following schema fields: `kan_proof_carrying_monitor_v2_ready`,
`kan_code_present`, `monitor_record_schema`, `attached_monitor_record_count`,
`local_error_bound_summary`, `global_error_bound_summary`,
`milp_property_check_count`, `false_accept_relevance`,
`deployed_verifier_claim`, `implementation_blockers`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

Each attached monitor record MUST carry the exact monitor fixture link,
piecewise-affine abstraction parameters, local and global error-bound
summaries, one MILP-compatible property result, and a deterministic checksum.
If KAN/PWA code or monitor fixture evidence is missing, the artifact MUST fail
closed with exact missing modules, schemas, tests, or source artifacts. The
artifact MUST NOT claim trained-network soundness, generation-path integration,
hardware execution, live LLM inference, model-weight mutation, or deployed
verifier improvement.

### SCENARIO-KAN-3145: Tiny Proof-Carrying Monitor Records Are Attached

Given the completed Exp 3131 KAN PWA/MILP abstraction audit and the Exp 3126
fragment-time monitor artifact,
When the Exp 3145 boundary builder runs on CPU,
Then it attaches proof-carrying KAN monitor records to the known `.291`
false-accept fixture subset when those fixtures are available, validates that
each record contains a fixture link, PWA abstraction parameters, local/global
error bounds, a MILP property result, and a checksum, reports whether the
records would have prevented or only audited the `.291` false-accept families,
and writes every required schema field with `deployed_verifier_claim=false`.

## REQ-KAN-3159: KAN Proof-Carrying Monitor Expansion V1

The KAN verification tier SHALL provide a bounded Exp 3159 expansion that
loads the prior Exp 3145 proof-carrying monitor records, the Exp 3131 KAN
PWA/MILP abstraction evidence, and the Exp 3136 exact false-accept/clean row
sets, then attaches proof-carrying metadata to a small number of additional
exact monitor rows when local code and artifacts support it.

The expansion MUST write
`results/experiment_3159_kan_proof_carrying_monitor_expansion_v1.json` with
the following schema fields: `kan_proof_carrying_monitor_expansion_v1_ready`,
`monitor_record_count`, `new_monitor_record_count`,
`exact_row_coverage_count`, `pwa_milp_bound_records`,
`deployed_verifier_claim_allowed`, `residual_blockers`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

Each `pwa_milp_bound_records` entry MUST expose a fixture ID, an exact-label
link, domain bounds, local and global PWA error-bound summaries, MILP/PWA
solver status, residual risk, and a deterministic checksum. The artifact MUST
set `deployed_verifier_claim_allowed=false` unless a real deployed verifier has
been implemented and tested. Bounded monitor-record expansion MUST NOT claim
trained-network soundness, live LLM inference, generation-path integration,
hardware execution, model-weight mutation, or deployment.

### SCENARIO-KAN-3159: Exact Clean Rows Extend Prior Monitor Records

Given the completed Exp 3145 boundary artifact and the Exp 3136 autopsy row
sets,
When the Exp 3159 expansion builder runs on CPU,
Then it carries forward the prior false-accept proof records, attaches new
records to additional clean exact rows, reports explicit total/new/exact-row
coverage counts, preserves inspectable PWA/MILP bound records, records residual
deployment blockers, and writes every required schema field with
`deployed_verifier_claim_allowed=false`.

## REQ-KAN-1384: EBM-CoT Hinge Calibration Probe on FoVer Pairs

The KAN energy tier SHALL support a CPU-only FoVer calibration probe that
warm-starts from a compatible local KAN checkpoint, trains for a fixed 20-epoch
budget with EBM-CoT's contrastive hinge objective, and reports held-out AUROC
before and after calibration.

The training objective MUST use low-energy-correct / high-energy-incorrect
semantics:

```
L = max(0, margin - (E_negative - E_positive))
    + lambda_consistency * |E_positive - E_positive_paraphrase|
```

**Acceptance criteria:**
    - The probe uses existing FoVer labeled step pairs and does not run fresh LLM inference.
    - The artifact includes `baseline_auroc`, `ebm_cot_auroc`,
      `calibration_auroc_delta`, `consistency_regularization_effect`, and
      `implicit_cot_energy_viable`.
    - `implicit_cot_energy_viable` is true iff `calibration_auroc_delta > 0`.
    - The probe records an honest verdict when no compatible checkpoint is available.

### SCENARIO-KAN-1384: FoVer hinge calibration artifact

Given a balanced FoVer split with correct and incorrect step labels,
When the EBM-CoT KAN calibration probe trains for 20 CPU epochs,
Then the output artifact contains the required calibration fields and computes
`calibration_auroc_delta = ebm_cot_auroc - baseline_auroc`.

## REQ-KAN-1401: EBM-CoT V2 Hinge-Only Calibration Probe

The KAN energy tier SHALL support an EBM-CoT v2 CPU-only calibration probe that
reuses the Exp1384 FoVer split and compatible KAN checkpoint while disabling
positive paraphrase consistency regularization.

The training objective MUST be hinge-only:

```
L = max(0, margin - (E_negative - E_positive))
```

The probe MUST write `results/experiment_1401_ebm_cot_v2_hinge_only.json` with
`consistency_regularization_weight=0.0`, `ebm_cot_v2_auroc`,
`calibration_auroc_delta`, `paraphrase_energy_variance_before`,
`paraphrase_energy_variance_after`, `variance_worsened`, and
`implicit_cot_energy_viable`.

**Acceptance criteria:**
    - The probe loads `baseline_auroc` from the Exp1384 artifact so the v2 delta
      is anchored to the prior result.
    - `variance_worsened` is true iff paraphrase energy variance after
      hinge-only training is greater than before.
    - `implicit_cot_energy_viable` is true iff `calibration_auroc_delta > 0`.
    - The artifact records an honest verdict describing whether hinge-only
      training confirms the positive calibration signal without consistency
      regularization.

### SCENARIO-KAN-1401: Hinge-only artifact gates v2 viability

Given the Exp1384 baseline artifact and the same FoVer train/test split,
When the EBM-CoT v2 probe trains with `consistency_regularization_weight=0.0`,
Then the output artifact uses `ebm_cot_v2_auroc` for the post-training score,
computes `calibration_auroc_delta = ebm_cot_v2_auroc - baseline_auroc`, and
sets `variance_worsened` from the measured paraphrase variance comparison.

## REQ-KAN-1819: KAN Verifier Latency Benchmark

The KAN model tier MUST provide a latency benchmark to measure the performance overhead of the CIKAN verifier against a baseline language model (e.g., `unsloth/gemma-4-31B-it-GGUF`).

**Rationale:**
    To deploy KAN verifiers in a production pipeline, we must verify that the latency overhead per token is within acceptable limits compared to the base model.

**Acceptance criteria:**
    - `scripts/experiment_1819_kan_latency.py` writes `results/experiment_1819_kan_latency.json`.
    - The JSON contains `baseline_tps`, `cikan_tps`, and `latency_overhead_percent`.
    - Tests verify the benchmark generation.

### SCENARIO-KAN-1819: Measure KAN Verifier Latency

Given a CIKAN verifier and a mock language model pipeline,
When the benchmark script is run,
Then it outputs the token-per-second (TPS) for both baseline and CIKAN, and `results/experiment_1819_kan_latency.json` is generated.

## REQ-KAN-5080: Tiny KAEM PWA/MILP Bridge Experiment

The KAN verification tier SHALL provide an Exp 5080 CPU-only diagnostic bridge
from an existing `UnivariateKAEMLayer` energy head to a piecewise-affine (PWA)
abstraction and a bounded mixed-integer linear property check. The experiment
MUST build the PWA abstraction from the KAEM layer's real knot/control-point
structure, record the abstraction error bound, and use an available local
linear-integer solver before claiming the property was checked.

The experiment MUST write
`results/experiment_5080_kan_pwa_milp_bridge_v466.json` with these top-level
fields: `honest_verdict`, `duration_s`, `inference_substrate`,
`kan_component_path`, `pwa_abstraction_built`, `milp_solver_available`,
`property_checked`, `property_holds`, `error_bound`, `binary_variable_count`,
`blocked_reason`, and `flagged_adversarial`. The artifact MUST set
`inference_substrate="deterministic_formal_check"` and MUST use a terminal
verdict prefix such as `success_kan_pwa_milp_property_verified_tiny` when a
solver proves the property or `blocked_kan_pwa_milp_solver_unavailable` when no
solver dependency is available.

The experiment MUST NOT claim a production KAN verifier, trained-network
soundness, hardware execution, or general MILP scalability. If the solver is
unavailable, it MUST emit the blocked artifact instead of silently substituting
an enumeration-only proof.

### SCENARIO-KAN-5080: Tiny KAEM Bound Property Uses Solver Or Blocks

Given a deterministic one-variable `UnivariateKAEMLayer` with monotone control
points on `[-1, 1]`,
When the Exp 5080 bridge builds the exact PWA segment representation and checks
the bound property `energy(x) <= 1.0`,
Then the artifact records the KAEM component path, a built PWA abstraction, the
solver availability decision, the binary variable count for segment selection,
the zero exact-PWA error bound with methodology note, and either a successful
solver certificate or a blocked solver-dependency reason.

## REQ-KAN-5091: Small Multi-Unit KAEM PWA/MILP Scale Telemetry

The KAN verification tier SHALL extend the Exp 5080 KAEM/PWA/MILP bridge to a
small two-input additive KAEM property without invoking an LLM. The experiment
MUST build piecewise-affine abstractions from real `UnivariateKAEMLayer`
control points, declare local and global abstraction error budgets, encode the
property through a deterministic local MILP/Z3 solver path when available, and
make solver complexity visible.

The experiment MUST write
`results/experiment_5091_kan_pwa_milp_scale_v467.json` with these top-level
fields: `honest_verdict`, `duration_s`, `inference_substrate`,
`abstraction_built`, `solver_available`, `property_statement`,
`property_status`, `property_holds`, `binary_variable_count`,
`constraint_count`, `pwa_piece_count`, `local_error_bound`,
`global_error_bound`, `solve_time_s`, `scale_blocker`, and
`flagged_adversarial`. The artifact MUST set
`inference_substrate="deterministic_formal_solver"` and MUST use a terminal
verdict such as `success_kan_pwa_milp_scale_property_verified_small` when the
small scale-up is proved, or
`complete_kan_pwa_milp_scale_blocked_by_solver_complexity` when solver
complexity prevents the proof.

The artifact MUST also report bound tightness and enough PWA/MILP structure to
distinguish this from the one-variable Exp 5080 diagnostic. It MUST NOT claim a
production KAN verifier, trained-network soundness, hardware execution, live
LLM inference, or broad MILP scalability.

### SCENARIO-KAN-5091: Two-Input KAEM Scale Property Reports Solver Counts

Given a deterministic two-variable `UnivariateKAEMLayer` with monotone control
points on `[-1, 1]^2`,
When the Exp 5091 bridge builds exact PWA segment representations for both
variables and checks the bound property for their additive KAEM energy,
Then the artifact records local/global error budgets, a two-input property
statement, PWA piece count, binary variable count, constraint count, solve time,
bound tightness, solver availability, property status, and a success or blocked
terminal verdict without any live LLM inference claim.

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-KAN-003 | Proposed | Exp 724 target: 3000-example balanced dataset |
| REQ-KAN-004 | Proposed | Exp 724 target: 16 knots/spline in v3 KAN |
| REQ-KAN-VERIFY-001 | Implemented | Exp 972 KAN MILP formal verification helpers; Exp 992 violation fix reuses the same checks. |
| REQ-KAN-020 | Implemented | Exp 866: LUT analysis complete. N=8 MLP-bound=14400 > 7680 (over budget). ISING_PRIORITY. |
| REQ-MODEL-SOS-001 | Implemented | Exp 1047: SOSKANEnergy confirmed 0 violations on 16000 test points. |
| REQ-KAN-1148 | Proposed | Exp 1148 target: MetaCluster-style centroid compression for SOSKANEnergyV3 with AUROC drop <= 0.02 and >=5x shrink. |
| REQ-KAN-1162 | Implemented | Exp 1162: KANELE-style Q8 LUT blueprint and hardware-complexity artifact generated for the compressed SOSKANEnergyV3 shape. |
| REQ-KAN-1174 | Proposed | Exp 1174 target: BiKA multiply-free complexity analysis for SOSKANEnergyV3, MetaCluster, and AMD XDNA NPU feasibility. |
| REQ-KAN-1199 | Proposed | Exp 1199 target: KANtize-style 8-bit/4-bit SOSKANEnergyV3 spline quantization with endpoint-sensitive precision and safetensors export. |
| REQ-KAN-1266 | Proposed | Exp 1266 target: deterministic QuantKAN-style 3-bit PTQ simulation plus LUT-KAN latency comparison for SOSKANEnergyV3. |
| REQ-KAN-1319 | Proposed | Exp 1319 target: hardware-portability audit only for local KAN verifier/repair candidates with no FPGA or analog execution claim. |
| REQ-KAN-1502 | Proposed | Exp 1502 target: no-synthesis KAN hardware accounting comparing naive, QuantKAN-like, and KAEM-style variants for current verifier components. |
| REQ-KAN-1516 | Implemented | Exp 1516 normalized KAN/KAEM proxy and model shapes into an explicit provenance manifest before any future synthesis claim. |
| REQ-KAN-1602 | Implemented | Exp 1602 exact-rational KAN forward pass uses Python `fractions.Fraction` for bit-identical formal-verification arithmetic. |
| REQ-KAN-1671 | Proposed | Exp 1671 target: no-synthesis Hybrid Zeckendorf rational CPU simulation and complexity/bounding certificate artifact for RKAN tiers. |
| REQ-KAN-1604 | Implemented | Exp 1604 Sparse KAN clustering records Global Group Lasso, spectral regularization, sparsity, and memory-compression metrics. |
| REQ-KAN-1648 | Implemented | Exp 1648 spectral constraint grouping layers Laplacian row groups on Sparse KAN centroid compression and records direct `compression_ratio`. |
| REQ-KAN-1618 | Proposed | Exp 1618 target: model-level PWA KAN wrapper for logical affine activation bounds over arbitrary 1D spline callables. |
| REQ-KAN-1623 | Implemented | Exp 1623 no-synthesis LUT and logic-depth accounting compares KANELÉ LUT mapping with KV260 Ising v3. |
| REQ-KAN-1384 | Proposed | Exp 1384 target: EBM-CoT contrastive hinge calibration on FoVer verified pairs. |
| REQ-KAN-1401 | Proposed | Exp 1401 target: EBM-CoT v2 hinge-only calibration on the Exp1384 FoVer split with consistency weight 0.0. |
| REQ-KAN-1690 | Proposed | Exp 1690 target: GloroKAN-style local Lipschitz forward-pass bounds for rational KArAt attention. |
| REQ-KAN-1723 | Implemented | Exp 1723: FourierCSP constraints compile into fixed CIKAN architectural boundaries with toy artifact evidence. |
| REQ-KAN-1749 | Implemented | Exp 1749: Symbolic-KAN routing layer embeds discrete primitive choices as tensor gates over learned scalar projections. |
| REQ-KAN-2005 | Proposed | Exp 2005 target: adaptive KAEM/KAN spline mesh refinement emits structural-change metrics and a completed artifact. |
| REQ-KAN-2876 | Implemented | Exp 2876 corrigendum separates local/global bounds and reports the local Z3 solver path or blocked-solver fallback. |
| REQ-KAN-2893 | Proposed | Exp 2893 target: no-hardware-claim RM/BOP/NABS accounting for the clean Exp 2876 tiny PWA/MILP fixture. |
| REQ-KAN-5080 | Proposed | Exp 5080 target: tiny KAEM PWA/MILP bridge artifact with success-or-blocked solver boundary. |
| REQ-KAN-5091 | Proposed | Exp 5091 target: two-input KAEM PWA/MILP scale telemetry with solver counts, error budgets, and bound tightness. |

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

## REQ-KAN-1671: Hybrid Zeckendorf Exact-Rational RKAN Audit

The repository SHALL provide a CPU-only, no-synthesis audit script for
Exact-Rational KAN tiers that replaces mock floating-point KAN arithmetic with
Hybrid Zeckendorf rational arithmetic. The audit MUST:

- define `scripts/experiment_1671_rkan_audit.py`;
- evaluate a deterministic mock KAN using exact rational arithmetic, rejecting
  implicit Python floats at the audit boundary;
- expose Hybrid Zeckendorf certificates for rational numerators and
  denominators so every reported value has an integer-decomposition witness;
- report operation-count complexity for edge, bias, interpolation, addition,
  and multiplication work;
- report deterministic bounding certificates for spline outputs and simulated
  sample energies; and
- write `results/experiment_1671_rkan.json` with terminal `status="complete"`
  only when the no-synthesis, exact-arithmetic, and certificate gates pass.

The script MUST NOT claim FPGA, ASIC, analog, or other hardware synthesis
validation. The output is an accounting pass and CPU simulation artifact only.

### SCENARIO-KAN-1671: Exact-rational audit emits bounded certificate artifact

Given the deterministic mock RKAN tier fixture,
When `scripts/experiment_1671_rkan_audit.py` runs on CPU,
Then `results/experiment_1671_rkan.json` records `float_operations_used=false`,
`hardware_synthesis_claimed=false`, operation complexity counts, Hybrid
Zeckendorf witness decompositions, and bounding certificates whose exact
sample-energy bounds contain every simulated exact energy.

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

## REQ-KAN-1266: QuantKAN 3-bit PTQ and LUT-KAN simulation

Experiment 1266 MUST extend the Exp 1199 SOSKANEnergyV3 quantization report with
a deterministic QuantKAN-style 3-bit post-training-quantization simulation and
a LUT-KAN inference-latency comparison. The artifact MUST be derived from the
FoVer v5 evaluation corpus and the Exp 1199 4-bit baseline when available, while
falling back to the documented SOS-KAN references when a legacy field is absent.

**Rationale:**
    Exp 1199 established the 4-bit KANtize reference for SOSKANEnergyV3 edge
    deployment. Ultra-edge NPU planning needs a 3-bit estimate and a separate
    lookup-table inference comparison before spending hardware effort. This is
    explicitly a simulation: the 3-bit AUROC is a deterministic GPTQ-style drop
    from the loaded 4-bit AUROC, and LUT-KAN latency is an analytical comparison
    of direct spline evaluation against a 256-point INT8 lookup table.

**Acceptance criteria:**
    - The runner SHALL read `results/fover_corpus_v5.json` and evaluate the
      first 200 pairs for artifact provenance.
    - The runner SHALL read the Exp 1199 artifact, accepting both legacy
      `quantized_auroc`/`model_size_mb` aliases and current
      `soskan_4bit_auroc`/`soskan_4bit_size_mb` fields.
    - The artifact SHALL include `auroc_curve` with
      `full_precision`, `8bit_ptq`, `4bit_ptq`, `3bit_ptq`, and `3bit_lut`.
    - The artifact SHALL include `quantkan_3bit_auroc`, `lut_kan_speedup`, and
      an `honest_verdict` formatted as
      `quantkan_3bit_auroc_X.XXXX_lut_speedup_X.Xx`.
    - The 3-bit model size SHALL be 75% of the loaded 4-bit model size, and the
      LUT overhead SHALL equal `n_vars * n_grid_points` INT8 bytes.
    - `scripts/experiment_1266_quantkan_3bit_lut_kan.py` SHALL write
      `results/experiment_1266_quantkan_3bit_lut_kan.json` with the required
      schema fields.

### SCENARIO-KAN-1266: write QuantKAN 3-bit plus LUT-KAN artifact

Given the FoVer v5 corpus and the Exp 1199 SOSKANEnergyV3 4-bit result,
When `scripts/experiment_1266_quantkan_3bit_lut_kan.py` runs,
Then `results/experiment_1266_quantkan_3bit_lut_kan.json` is written with the
AUROC curve, 3-bit AUROC, LUT-KAN speedup, model-size comparison, and honest
verdict required by REQ-KAN-1266.

## REQ-KAN-1319: KAN hardware portability audit artifact

Experiment 1319 MUST audit at least one local KAN verifier or repair-adjacent
candidate for hardware portability without claiming real FPGA, NPU, or analog
execution unless local hardware execution actually occurred during the task.

**Rationale:**
    The KAN hardware and analog literature motivates more portable KAN
    representations, but Carnot needs a conservative local accounting step
    before any deployment claim. The audit should connect the active local
    SOS-KAN/QuantKAN/KANELE/BiKA evidence to transparent estimates for read
    memory, bit operations, nonlinear activation budget, lookup-table memory,
    and near-term target platform.

**Acceptance criteria:**
    - `results/experiment_1319_kan_hardware_complexity_audit.json` is written
      with `run_date="20260505"`.
    - The artifact includes the fields `status`, `rm_per_inference`,
      `bop_per_inference`, `nabs_per_inference`, `lookup_table_bytes`,
      `analog_kan_candidate`, `npu_or_fpga_best_target`,
      `hardware_claim_allowed`, and `honest_verdict`.
    - The audit names the local KAN or repair-adjacent modules it considered.
    - `hardware_claim_allowed` is false unless real local hardware execution
      occurred during the experiment.
    - The honest verdict MUST state that the result is a hardware-portability
      audit, not FPGA, NPU, or analog execution, when no hardware execution
      occurred.

### SCENARIO-KAN-1319: write conservative KAN hardware complexity audit

Given the local Exp 1148/1162/1174 SOSKANEnergyV3 compression and complexity
artifacts,
When `scripts/experiment_1319_kan_hardware_complexity_audit.py` runs,
Then `results/experiment_1319_kan_hardware_complexity_audit.json` is written
with deterministic RM/BOP/NABS/LUT-memory estimates, platform classification,
`hardware_claim_allowed=false`, and an honest non-execution verdict.

## REQ-KAN-1502: no-synthesis KAN hardware accounting for verifier components

Experiment 1502 MUST produce a no-synthesis hardware-accounting artifact for
Carnot-relevant KAN verifier components. The artifact MUST compare a naive
full-precision KAN/SOS-KAN path, a QuantKAN-like quantized lookup-table path,
and a KAEM-style univariate approximation path using only accounting evidence
from local source files and prior artifacts.

**Rationale:**
    Post-.114 research references added KAN hardware-complexity and ultra-light
    accelerator signals, but Exp 1460 narrowed the active hardware portfolio and
    forbids new KAN accelerator claims without synthesis, board, or measured
    hardware evidence. Carnot still needs transparent operation, memory, LUT,
    BRAM, and accuracy-risk estimates so future hardware work can stay scoped.

**Acceptance criteria:**
    - `results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json` is
      written with `run_date="20260507"` and `status="complete"`.
    - The artifact includes `status`, `kan_hardware_accounting_ready`,
      `accounting_only_no_synthesis_claim`, `kan_components_audited`,
      `quantkan_proxy_estimates`, `kaem_proxy_estimates`,
      `lut_proxy_estimate`, `bram_proxy_estimate`, `accuracy_risk_notes`,
      `hardware_claim_allowed`, `blockers`, and `honest_verdict`.
    - `hardware_claim_allowed` is false and
      `accounting_only_no_synthesis_claim` is true unless actual synthesis or
      board measurement evidence is present.
    - The accounting table reports operation counts, memory footprints,
      rough LUT/BRAM proxy pressure, and accuracy-risk boundaries for naive,
      QuantKAN-like, and KAEM-style variants.
    - `honest_verdict` starts with one of the conductor terminal prefixes:
      `complete:`, `complete_`, `success:`, `success_`, `passed:`,
      `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-KAN-1502: write conservative QuantKAN/KAEM accounting artifact

Given the local Exp 1148/1162/1174/1199/1266/1319/1372 KAN artifacts and the
current KAN verifier modules,
When the Exp 1502 accounting helper runs,
Then `results/experiment_1502_kan_hardware_accounting_quantkan_kaem.json`
contains the required fields, compares naive/QuantKAN-like/KAEM-style variants,
sets `hardware_claim_allowed=false`, and records a terminal no-synthesis honest
verdict.

## REQ-KAN-1516: KAN/KAEM shape normalization preflight

Experiment 1516 MUST normalize KAN/KAEM model and proxy dimensions into a
separate shape manifest before future hardware synthesis work can cite Exp 1502
accounting. The preflight MUST read the Exp 1502 hardware-accounting artifact,
verify the Exp 1506 prior blocker, map model/proxy dimensions to
hardware-accounting dimensions with explicit provenance, and record excluded
shape assumptions.

**Rationale:**
    Exp 1502 intentionally avoided Vivado, bitstreams, board execution, and
    timing closure. It also recorded that QuantKAN and KAEM proxy shapes must be
    normalized before future synthesis. That hygiene gate prevents later work
    from treating proxy read-memory, bit-operation, LUT, or BRAM counts as
    synthesis-ready shapes unless the exact source fields and exclusions are
    visible.

**Acceptance criteria:**
    - `results/experiment_1516_kan_shape_normalization_preflight.json` is
      written with `run_date="20260508"` and the required terminal fields:
      `status`, `kan_shape_manifest_ready`, `gated_inputs_present`,
      `no_synthesis_claim`, `no_board_claim`, `proxy_shapes_loaded`,
      `normalized_shapes_written`, `excluded_shape_assumptions`,
      `hardware_accounting_shape_fields`, `shape_manifest_path`, `blockers`,
      and `honest_verdict`.
    - The preflight verifies
      `results/experiment_1506_115_completion_archive_116_activation.json`
      has `prior_kan_shape_blocker_recorded=true`; otherwise it writes a
      terminal gated artifact and does not mark the manifest ready.
    - The normalized manifest records proxy dimensions, batch/sequence
      assumptions, quantization assumptions, and hardware-accounting dimensions
      with explicit artifact-field provenance for each mapped variant.
    - The terminal artifact sets `no_synthesis_claim=true` and
      `no_board_claim=true`; `kan_shape_manifest_ready` is true only when the
      normalized manifest and excluded assumptions are written.
    - `honest_verdict` starts with one of the conductor terminal prefixes:
      `complete:`, `complete_`, `success:`, `success_`, `passed:`,
      `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-KAN-1516: write normalized KAN shape manifest

Given the completed Exp 1502 KAN hardware-accounting artifact and the Exp 1506
prior blocker record,
When the Exp 1516 shape-normalization preflight runs,
Then `results/kan_shape_normalization_manifest_1516.json` maps at least the
naive SOS-KAN, QuantKAN LUT, and KAEM univariate variants to normalized
hardware-accounting shapes, records all excluded shape assumptions, blocks
synthesis and board claims, and writes a complete Exp 1516 artifact.

## REQ-KAN-1599: KANELÉ hardware LUT-complexity accounting

Experiment 1599 MUST perform a no-synthesis hardware accounting pass for KANs.
It MUST estimate RM, BOP, and NABS for KANs and write the artifact
`results/experiment_1599_kanele_audit.json` without claiming any actual
hardware synthesis or board execution.

**Rationale:**
    We need to track KANELÉ LUT complexity (Resource Metrics, Bit Operations,
    Number of Additions/Subtractions) for the current milestone without
    actually running Vivado or claiming board execution.

**Acceptance criteria:**
    - Calculates RM, BOP, and NABS based on some existing KAN architecture or artifact.
    - Writes `results/experiment_1599_kanele_audit.json` with fields `rm_per_inference`, `bop_per_inference`, and `nabs_per_inference`.
    - Makes absolutely no claims of synthesis or execution (`hardware_execution_confirmed: False`).
    - Contains an approved `honest_verdict`.

### SCENARIO-KAN-1599: write no-synthesis KAN accounting artifact

Given a KAN architecture or prior artifact,
When the Exp 1599 audit runs,
Then `results/experiment_1599_kanele_audit.json` is created with RM, BOP,
and NABS estimates, and `hardware_execution_confirmed: False`.

## REQ-KAN-1602: Exact-Rational KAN Forward Pass

The KAN model tier MUST provide an exact-rational CPU forward pass for formal
verification. The implementation MUST evaluate edge and bias spline
contributions entirely with Python `fractions.Fraction` values so repeated
passes over equivalent rational inputs produce identical numerator/denominator
results without floating-point rounding or JAX tracing.

**Rationale:**
    Formal verification needs arithmetic whose semantics are explicit and
    replayable. The JAX KAN path is appropriate for training and differentiable
    inference, but binary floating-point evaluation can obscure whether two
    proof runs differ because of model semantics or because of backend numeric
    details. A small exact-rational RKAN forward pass gives property tests and
    formal auditors a deterministic reference for KAN energy evaluation.

**Acceptance criteria:**
    - `python/carnot/models/rkan.py` exposes `RationalLinearSpline` and
      `RationalKANEnergyFunction`.
    - The RKAN forward pass accepts integers, strings, and `Fraction` inputs,
      converts them to `Fraction`, validates input shape and edge indices, and
      returns a `Fraction`.
    - Edge products, piecewise-linear interpolation, bias terms, and total
      energy accumulation use exact rational operations only.
    - Repeated forward passes over equivalent rational inputs produce identical
      `Fraction` outputs.
    - The experiment artifact `results/experiment_1602_rkan.json` is written
      with `schema`, `status`, `experiment_id`, `spec`,
      `exact_rational_forward_pass_ready`, `float_operations_used`,
      `repeated_forward_outputs_identical`, `sample_outputs`, and a terminal
      `honest_verdict`.

### SCENARIO-KAN-1602: exact-rational RKAN artifact and deterministic energy

Given a small RKAN with rational edge and bias spline control points,
When the exact-rational forward pass evaluates rational input vectors twice,
Then every intermediate contribution and output remains a `Fraction`, the
second pass is bit-identical to the first, and
`results/experiment_1602_rkan.json` records the completed deterministic
reference artifact.

## REQ-KAN-1604: Sparse KAN Clustering With Global Group Lasso

The KAN model tier MUST provide a CPU-only Sparse KAN clustering helper that
compresses constraint-memory rows by learning a small centroid codebook while
penalizing whole inactive groups with a Global Group Lasso term and preserving
cluster geometry with a spectral constraint regularizer.

The regularized loss MUST expose separate components:

```
L = reconstruction_loss
    + lambda_group_lasso * sum_g ||C_g||_2
    + lambda_spectral * trace(Z^T L Z)
```

where `C_g` is one centroid/control-vector group, `Z` is the row-to-cluster
assignment matrix, and `L` is a graph Laplacian derived from constraint-row
affinity.  The helper MUST report a deterministic sparsity ratio from the
number of zeroed centroid groups divided by the total group count.

**Acceptance criteria:**
    - `SparseKANClusterer.regularized_loss_components()` returns
      `reconstruction_loss`, `global_group_lasso_penalty`,
      `spectral_constraint_regularization`, and `total_loss`.
    - Applying the sparsifier with a deterministic threshold zeroes low-norm
      centroid groups and the result `sparsity_ratio` returns
      `zero_group_count / n_clusters`.
    - `fit()` produces finite centroids, deterministic assignments, and a
      compressed-memory estimate smaller than the dense constraint matrix.
    - `write_experiment_1604_artifact()` writes
      `results/experiment_1604_sparse_kan.json` with `schema`, `status`,
      `experiment_id`, `spec`, `sparse_kan_clustering_ready`,
      `global_group_lasso_penalty`, `spectral_constraint_regularization`,
      `sparsity_ratio`, `memory_compression_ratio`, and a terminal
      `honest_verdict`.

### SCENARIO-KAN-1604: sparse KAN clustering artifact records compression

Given a deterministic synthetic constraint-memory matrix with repeated
structure,
When Sparse KAN clustering fits a small centroid codebook with Global Group
Lasso and spectral regularization enabled,
Then the low-norm groups are pruned, the sparsity ratio is measured from the
active centroid codebook, the compressed-memory estimate is smaller than the
dense matrix, and `results/experiment_1604_sparse_kan.json` records the
completed compression artifact.

## REQ-KAN-1648: Spectral Constraint Grouping for Tier 4 Sparse KAN Landscapes

The KAN model tier MUST provide a CPU-only Exp 1648 probe that layers spectral
constraint grouping on top of the existing Sparse KAN centroid compressor for
Tier 4 adaptive landscapes.  The probe MUST derive a deterministic row-affinity
graph from adaptive-landscape observations, group rows in spectral embedding
space, compress each group through Sparse KAN centroids, and report both the
spectral grouping quality and memory compression.

The artifact MUST record the direct field `compression_ratio` in addition to
the inherited Sparse KAN memory accounting so the conductor can compare Exp
1648 against prior sparse-compression experiments without schema-specific
field names.

**Acceptance criteria:**
    - `scripts/experiment_1648_sparse_kan.py` exposes deterministic helpers for
      building a Tier 4 adaptive-landscape matrix, spectral grouping rows, and
      computing grouped Sparse KAN compression metrics.
    - Spectral grouping uses a graph Laplacian embedding or equivalent
      eigenvector-derived ordering before assigning rows to groups.
    - The artifact includes `schema`, `status`, `experiment_id`, `spec_traces`,
      `n_constraint_rows`, `n_spectral_groups`, `spectral_gap`,
      `spectral_grouping_penalty`, `dense_memory_bytes`,
      `compressed_memory_bytes`, `compression_ratio`, and `honest_verdict`.
    - `compression_ratio` is exactly
      `dense_memory_bytes / max(compressed_memory_bytes, 1)`.
    - A completed result is written to
      `results/experiment_1648_sparse_kan.json` with a terminal
      `honest_verdict`.

### SCENARIO-KAN-1648: spectral grouping records compression ratio

Given a deterministic adaptive-landscape matrix with repeated local structures,
When Exp 1648 builds spectral groups and compresses them with Sparse KAN
centroids,
Then high-affinity rows remain grouped, the spectral grouping penalty is finite,
the compressed-memory estimate is smaller than the dense matrix, and
`results/experiment_1648_sparse_kan.json` records the direct
`compression_ratio` field.

## REQ-KAN-1618: PWA KAN Wrapper for Logical Spline Activation Bounds

The KAN model tier MUST provide a CPU-only Piecewise Affine (PWA) wrapper for
one-dimensional KAN spline activations. The wrapper MUST support arbitrary 1D
spline callables by sampling them at declared breakpoints, computing affine
center lines, and deriving conservative affine lower and upper activation
boundaries for each segment.

**Rationale:**
    Formal verification workflows need KAN activations to be expressible as
    local linear implications: if an input lies in one segment, the activation
    lies between two affine boundary functions. Exact knot-aligned linear KAN
    splines should have zero envelope error, while nonlinear or externally
    supplied spline callables still need deterministic sampled envelopes that
    are safe for logical downstream checks.

**Acceptance criteria:**
    - `python/carnot/models/pwa_kan.py` exposes a PWA spline wrapper for
      arbitrary 1D callables and existing `BSpline` KAN units.
    - Each segment records `slope`, `intercept`, affine lower/upper boundaries,
      residual bounds, and `max_abs_error`.
    - Interval activation-bound queries check every overlapping segment and
      return deterministic lower/upper values with witness inputs.
    - The wrapper can emit JSON-safe logical constraints of the form
      `x in [lo, hi] => lower_affine(x) <= y <= upper_affine(x)`.
    - The experiment artifact `results/experiment_1618_pwa_kan.json` is written
      with `schema`, `status`, `experiment_id`, `spec`, logical-bound metrics,
      and a terminal `honest_verdict`.

### SCENARIO-KAN-1618: PWA wrapper bounds exact and nonlinear spline units

Given an exact piecewise-linear KAN spline and a nonlinear 1D spline callable,
When the PWA wrapper builds affine segment boundaries and evaluates interval
activation bounds,
Then the exact KAN spline has zero sampled envelope error, nonlinear samples are
contained inside their segment lower/upper affine envelopes, and
`results/experiment_1618_pwa_kan.json` records the completed model-level PWA
KAN abstraction artifact.

## REQ-KAN-1621: KANELE Python-to-Verilog 6-input LUT synthesis

The KAN capability MUST include a Python-to-Verilog compiler that translates 1D
edge functions directly into configuration bits for 6-input LUTs.

**Rationale:**
    To synthesize KAN edges into FPGA fabric, we need to map generic 1D
    functions directly to 6-input LUT initializations (INIT parameters) rather
    than relying on Vivado to infer tables.

**Acceptance criteria:**
    - A Python script `python/carnot/hardware/kan_lut_compiler.py` (or similar)
      can take a generic 1D Python function `f(x)` over 6-bit input and produce
      a 64-bit Verilog LUT INIT string.
    - The compiler generates `hardware/kv260/kan_lut_block.v` containing
      instantiated `LUT6` primitives.
    - The experiment artifact `results/experiment_1621_kanele_mapping.json` is
      written with `schema`, `status`, `experiment_id`, `spec`,
      `kan_lut_verilog_ready`, `lut_config_bits_generated`,
      `kan_lut_block_written`, and a terminal `honest_verdict`.

### SCENARIO-KAN-1621: Generates KAN LUT Verilog module

Given a simple 1D KAN edge function,
When the LUT compiler translates it into 6-input LUT configuration bits,
Then `hardware/kv260/kan_lut_block.v` is generated with the correct INIT values,
and `results/experiment_1621_kanele_mapping.json` records the completion.

## REQ-KAN-1623: KANELÉ vs Ising v3 LUT and Logic-Depth Accounting

Experiment 1623 MUST produce a no-synthesis KV260 hardware-bounds accounting
artifact comparing the KANELÉ LUT-mapped edge datapath from Exp 1621 with the
existing `hardware/kv260/ising_sampler_v3.v` formulation. The accounting MUST
calculate per-node LUT consumption, estimate maximum clock frequency from
logic-depth assumptions, and keep synthesis and board-execution claims false.

**Rationale:**
    Exp 1621 generated direct LUT6 Verilog for a KANELÉ-style 1D edge block.
    Before requesting Vivado synthesis or KV260 board time, Carnot needs a
    transparent resource and critical-path estimate against the Ising v3 RTL
    baseline already present under `hardware/kv260/`.

**Acceptance criteria:**
    - `results/experiment_1623_kanele_accounting.json` is written with
      `schema`, `status`, `experiment_id`, `spec`, `per_node_lut_consumption`,
      `logic_depth_estimate`, `max_clock_frequency_estimate_mhz`,
      `kv260_budget`, `hardware_claim_allowed`, and `honest_verdict`.
    - The KANELÉ per-node LUT count is derived from the number of `LUT6`
      primitives in `hardware/kv260/kan_lut_block.v` plus explicit control and
      accumulation assumptions.
    - The Ising v3 per-node LUT count is derived from the documented KV260
      utilization comment in `hardware/kv260/ising_sampler_v3.v` and the
      XCK26 LUT budget used by the existing FPGA spec.
    - The maximum clock-frequency estimates use stated LUT-delay and
      register-overhead assumptions and do not claim timing closure.
    - `hardware_claim_allowed=false`, `synthesis_performed=false`, and
      `board_execution_performed=false` unless the current run records actual
      synthesis or KV260 board evidence.

### SCENARIO-KAN-1623: write KANELÉ vs Ising accounting artifact

Given `hardware/kv260/kan_lut_block.v`, the Exp 1621 mapping artifact, and
`hardware/kv260/ising_sampler_v3.v`,
When the Exp 1623 accounting helper runs,
Then `results/experiment_1623_kanele_accounting.json` records deterministic
per-node LUT, logic-depth, and maximum-clock estimates for KANELÉ and Ising v3
while preserving the no-synthesis/no-board hardware-claim boundary.

## REQ-KAN-1637: KANELÉ RTL Vivado Linting Preflight

The KAN capability MUST perform a preflight check for Vivado installation and attempt linting on KANELÉ RTL (`hardware/kv260/kan_lut_block.v`).

**Rationale:**
    RTL linting was blocked due to missing Vivado paths. We need to explicitly check if Vivado is available in the environment before attempting to run `xvlog` or `vivado`, and output the results to a structured artifact.

**Acceptance criteria:**
    - `scripts/experiment_1637_lint.py` is written and executable.
    - `results/experiment_1637_vivado_lint.json` contains `vivado_installed` and `lint_passed`.
    - Unit tests exist for the preflight script logic and reference `REQ-KAN-1637`.

### SCENARIO-KAN-1637: KANELÉ RTL Vivado Linting Preflight

Given the `hardware/kv260/` directory,
When `scripts/experiment_1637_lint.py` is executed,
Then it checks for Vivado, attempts linting if possible, and outputs `vivado_installed` and `lint_passed` to the JSON artifact.

## REQ-KAN-1688: CIKAN Regularizer for Monotonic B-splines

The KAN model tier MUST provide a CIKAN regularizer that enforces monotonic behavior on B-spline coefficients directly, without requiring post-hoc projection.

**Rationale:**
    B-spline monotonicity can be enforced by constraining differences between adjacent coefficients. A regularizer that penalizes non-monotonic coefficient sequences allows continuous gradient-based optimization while encouraging the spline to become monotonically increasing or decreasing.

**Acceptance criteria:**
    - `python/carnot/models/cikan_reg.py` implements `CIKANRegularizer`.
    - Tests verify that the regularizer computes the correct penalty for non-monotonic coefficients.
    - A test spline's non-monotonic behavior is penalized correctly.
    - `results/experiment_1688_cikan.json` is written with the appropriate results.

### SCENARIO-KAN-1688: CIKAN Regularizer penalizes non-monotonicity

Given a set of B-spline coefficients,
When `CIKANRegularizer` is applied,
Then it computes a penalty proportional to the non-monotonic adjacent differences, and `results/experiment_1688_cikan.json` records the success.

## REQ-KAN-1723: FourierCSP-Boundary CIKAN Verifier

The KAN model tier MUST provide a `CIKAN` verifier that accepts constraints
emitted by `FourierCSPExtractor` and compiles them into fixed architectural
boundary units before any residual KAN training occurs.

The boundary units SHALL:

- preserve the FourierCSP variable names, expression text, and polynomial text;
- evaluate the supported Boolean operators `AND`, `OR`, `NOT`, and `XOR` over
  thresholded feature values;
- contribute a fixed violation penalty to the verifier energy/logit path; and
- remain immutable across `fit()` calls so gradient updates cannot move or
  remove physical/logical constraint boundaries.

The verifier MAY train a small residual KAN head on top of those fixed
boundaries, but a violated FourierCSP constraint MUST retain higher energy than
an otherwise matching satisfying assignment after training.

**Acceptance criteria:**
    - `python/carnot/models/cikan_verifier.py` exposes `CIKAN` and
      `CIKANBoundary`.
    - Tests show FourierCSP constraints compile into fixed architectural
      boundaries and remain unchanged after training.
    - Tests show a violating assignment has higher CIKAN energy than a
      satisfying assignment for the same FourierCSP constraint.
    - `scripts/experiment_1723_cikan.py` trains on a deterministic toy dataset
      and writes `results/experiment_1723_cikan.json` with artifact schema,
      constraint, training, metric, and verdict fields.

### SCENARIO-KAN-1723: FourierCSP constraints remain fixed during CIKAN training

Given a FourierCSP multilinear polynomial for `X AND Y`,
When the CIKAN verifier compiles it and trains on a toy dataset,
Then its boundary snapshot before and after training is identical, the violated
assignment has higher energy than the satisfying assignment, and the Exp 1723
artifact reports `fixed_boundaries_preserved=true`.

## REQ-KAN-1749: Symbolic-KAN Primitive Routing Tensor Embedding

The KAN model tier MUST provide a CPU/JAX prototype of the arXiv:2603.23854
Symbolic-KAN mapping in which each route learns a scalar projection of the
input tensor, evaluates a finite library of analytic primitives on that
projection, and embeds the route's discrete symbolic structure as a gate tensor
over the primitive library.

The prototype SHALL:

- keep the symbolic primitive set finite, named, and deterministic;
- compute soft primitive gates from route logits and temperature;
- support hard one-hot gate extraction for discretized symbolic structures;
- return a structured forward-pass report with projections, primitive values,
  gates, route values, and energies; and
- expose a symbolic regularization value that decreases as gates sharpen toward
  one-hot selections.

**Acceptance criteria:**
    - `python/carnot/models/kan/symbolic_kan.py` exposes
      `SymbolicRoutingLayer`, `SymbolicKANConfig`, `SymbolicKANParams`,
      `build_experiment_1749_artifact`, and `write_experiment_1749_artifact`.
    - Unit tests verify soft routing, hard one-hot routing, batch/vector
      forward-pass shapes, selected primitive names, entropy-style symbolic
      regularization, validation failures, and stable artifact schema fields.
    - `results/experiment_1749_symbolic_kan.json` is written with the required
      schema fields and an honest verdict.

### SCENARIO-KAN-1749: Symbolic primitive routing maps to tensor space

Given a `SymbolicRoutingLayer` with two scalar projection routes and a finite
primitive library,
When the layer evaluates vector and batch inputs with soft and hard routing,
Then its gate tensor has shape `(n_routes, n_primitives)`, hard gates are
one-hot, selected primitive names match the maximum route logits, and the Exp
1749 artifact reports a complete Symbolic-KAN tensor-space prototype.

## REQ-KAN-1679: Miniature KArAt Attention Block

The KAN capability MUST provide a miniature KArAt attention block (Kolmogorov-Arnold Attention) that replaces Softmax with learnable spline/rational bases. The implementation MUST be designed for energy calculation, and its parameter counts and bounding bounds MUST be verified using rational abstractions.

**Rationale:**
    Moving beyond standard MLP and attention to fully learnable, verifiable bases (arXiv:2503.10632). Replacing Softmax with rational bases allows for exact verification and bounding, which is critical for formal property checks and deterministic energy calculations.

**Acceptance criteria:**
    - `python/carnot/models/karat_attention.py` exposes a single KArAt layer designed for energy calculation.
    - Parameter counts and bounding bounds are verified using rational abstractions.
    - Tests verify the model logic and achieve 100% test coverage.
    - `results/experiment_1679_karat.json` is written with the required schema fields, parameter counts, bounding bounds validation, and an honest verdict.

### SCENARIO-KAN-1679: KArAt attention block artifact

Given the `karat_attention.py` module,
When the parameter counts and bounding bounds are verified,
Then the tests pass, coverage is 100%, and `results/experiment_1679_karat.json` is written.

## REQ-KAN-1689: Certified KArAt Model Evaluation

The KAN capability MUST provide a script to evaluate the predictive performance of the MILP-certified KArAt model against an uncertified baseline on a synthetic reasoning dataset.

**Rationale:**
    To ensure the MILP-certified KArAt model maintains or improves accuracy and output bounds compared to the uncertified baseline.

**Acceptance criteria:**
    - `python/carnot/models/certified_karat.py` implements the certified KArAt wrapper and benchmark logic.
    - Tests verify the model logic and achieve 100% test coverage.
    - `results/experiment_1689_certified_karat.json` is written with the required schema fields, comparing accuracy and output bounds, and an honest verdict.

### SCENARIO-KAN-1689: Certified KArAt Evaluation

Given the `certified_karat.py` module,
When the benchmark script is run on the synthetic reasoning dataset,
Then accuracy and output bounds are compared, tests pass, coverage is 100%, and `results/experiment_1689_certified_karat.json` is written.

## REQ-KAN-1690: GloroKAN-style Local Lipschitz Bounds for KArAt

The KAN capability MUST provide a CPU-only GloroKAN-style forward-pass local
Lipschitz bound calculator for the miniature rational KArAt attention model.
The calculator MUST use spline control-point slopes, dot-product interval
bounds, and norm-bounded query/key magnitudes to return a deterministic upper
bound on how much the KArAt energy can change under a local input perturbation.

**Rationale:**
    GloroKAN-style robustness relies on a spline-specific observation: a
    forward pass does not need a black-box global neural-network Lipschitz
    estimate when each one-dimensional activation exposes knot-aligned local
    slopes. For KArAt, each attention term is
    `spline(q_i dot k_j)`, so the local bound can combine the maximum spline
    slope on the reachable dot-product interval with a conservative chain-rule
    bound for the local query and key vector norms.

**Acceptance criteria:**
    - `python/carnot/models/kan/glorokan_robustness.py` exposes
      `GloroKANBounder`.
    - The bounder accepts a `RationalKArAtLayer` and exact rational query/key
      inputs.
    - It returns a structured bound report with schema-safe fields including
      `local_lipschitz_bound`, `radius`, `norm`, per-term dot intervals, and
      per-term spline slope bounds.
    - The reported local Lipschitz bound is deterministic and is at least as
      large as a finite-difference energy-change witness inside the requested
      local radius.
    - `results/experiment_1690_glorokan_robustness.json` is written with the
      required schema fields and an honest verdict.

### SCENARIO-KAN-1690: KArAt local robustness report bounds a perturbation

Given a `RationalKArAtLayer` with rational spline control points and rational
query/key matrices,
When `GloroKANBounder.bound_forward()` is called with a positive local radius,
Then the report includes deterministic per-term dot intervals and a nonnegative
local Lipschitz bound that upper-bounds the observed KArAt energy change for a
same-radius perturbation.

## REQ-KAN-1729: KANELÉ CIKAN to FPGA LUT Mapping Pipeline

The KAN capability MUST provide an RTL generation pipeline that maps Continuous Interpretable Kolmogorov-Arnold Networks (CIKAN) to FPGA Lookup Tables (LUTs), creating a top-level Verilog wrapper and a Python-to-Verilog mapper.

**Rationale:**
To accelerate CIKANs on FPGA hardware (Phase 3), the trained Python model weights must be compiled directly into Verilog LUT definitions. This requires a dedicated mapper script and a top-level module to orchestrate the LUT blocks for deployment on the KV260 board.

**Acceptance criteria:**
- `hardware/kv260/kanele_lut_mapper.py` implements the CIKAN to FPGA LUT mapping logic.
- `hardware/kv260/kanele_top.v` provides the top-level Verilog wrapper.
- `scripts/experiment_1729_kanele.py` executes a simulation check and writes `results/experiment_1729_kanele.json`.
- Tests verify the generation pipeline logic.

### SCENARIO-KAN-1729: CIKAN to FPGA LUT Mapping

Given a CIKAN model,
When the LUT mapper and RTL generation pipeline run,
Then the correct Verilog files are generated, the simulation check passes, and `results/experiment_1729_kanele.json` records the success.

## REQ-KAN-1808: Symbolic Regression via CIKAN

The KAN capability MUST support extracting exact logical/algebraic formulas from a trained CIKAN layer.
By feeding a dataset (e.g., modeling Z = X + Y), the CIKAN layer must discover the correct symbolic operations
and allow comparison against the ground truth equation.

**Acceptance criteria:**
    - `python/carnot/models/carnot_kan/cikan_layer.py` exposes a `CIKANLayer` capable of symbolic extraction.
    - `scripts/experiment_1808_cikan_symbolic.py` fits a known dataset, extracts the symbolic equation, and logs `equation_match_accuracy`.
    - Unit tests cover the CIKAN layer.
    - `results/experiment_1808_symbolic.json` contains `equation_match_accuracy`.

### SCENARIO-KAN-1808: Extract symbolic equation correctly

Given a dataset matching an exact arithmetic rule (e.g., Z = X + Y),
When CIKANLayer is fitted to this dataset,
Then it extracts the correct symbolic equation, achieving high equation_match_accuracy,
and `results/experiment_1808_symbolic.json` is logged.

## REQ-KAN-1840: PWA KAN Abstraction

The KAN capability MUST implement Piecewise Affine (PWA) abstractions for nonlinear KAN units to support MILP verification (arXiv:2602.06737).

**Rationale:**
    Formal verification of KAN units requires converting continuous, nonlinear spline functions into piecewise-linear components that can be encoded in an MILP solver. The abstraction must compute affine bounds (lower and upper) for each linear segment.

**Acceptance criteria:**
    - `python/carnot/verify/pwa_kan.py` exposes a PWA abstraction function that converts 1D splines to piecewise-linear segments with bounds.
    - Each segment contains bounds information.
    - `results/experiment_1840_pwa_kan.json` is generated correctly.
    - Tests verify the logic and achieve 100% coverage, referencing `REQ-KAN-1840`.

### SCENARIO-KAN-1840: Spline to PWA conversion

Given a nonlinear KAN unit (e.g. spline callable),
When the PWA abstraction is applied,
Then it computes piecewise-linear approximations with affine bounds for each segment, tests pass with 100% coverage, and `results/experiment_1840_pwa_kan.json` is written.

## REQ-KAN-1826: KAN-CL Per-Knot Importance Regularization

The KAN capability MUST implement the KAN-CL algorithm (arXiv:2605.11181) for continual learning. This requires per-knot importance regularization to prevent catastrophic forgetting.

**Rationale:**
    Continuous self-learning suffers from catastrophic forgetting in KAN energy tiers.
    By tracking the importance weight of each B-spline knot during training,
    subsequent learning phases can apply a regularization penalty that anchors
    important knots while allowing unimportant knots to adapt to new tasks.

**Acceptance criteria:**
    - `python/carnot/models/kan_cl.py` implements the KAN-CL importance tracker and regularization penalty.
    - It tracks importance weights for B-spline knots during training.
    - It computes a penalty term for subsequent learning phases based on the deviation from anchored control points, weighted by the tracked importance.
    - `python/carnot/learning/kan_cl.py` exposes `KanClLearner.fit(X, y, task_id)` and `KanClLearner.predict(X)` for n=256 constraint learning.
    - The learner records per-knot importance as the activation frequency of each of the 256 spline coefficients for every task.
    - The split-task benchmark writes `results/experiment_2356_kancl_n256.json` and validates KAN-CL when catastrophic-forgetting reduction is at least 50%.
    - Tests verify that the penalty is computed correctly and achieve 100% test coverage for the new module.

### SCENARIO-KAN-1826: KAN-CL Penalty Computation

Given an initial set of B-spline control points and their corresponding importance weights,
When a subsequent learning phase proposes updated control points,
Then the KAN-CL regularization term correctly computes a penalty proportional to the importance-weighted deviation, and tests pass with 100% coverage.

### SCENARIO-KAN-1826-N256: KAN-CL Split-Task Constraint Learning

Given three 50-example constraint-learning tasks over arithmetic, code, and logic domains with `n_params=256`,
When `KanClLearner` trains sequentially on the tasks,
Then per-knot importance is computed from task activation frequencies, prior-task coefficients are protected by importance-weighted L2 updates, and the artifact reports `forgetting_reduction_pct >= 50`.


## REQ-KAN-1857: Softly Symbolified KANs (S2KAN) with Differentiable Gating

The KAN capability MUST implement Softly Symbolified KANs (S2KAN) that introduce symbolic primitives with differentiable gating.

**Rationale:**
    To enable verifiable KANs, the network needs to use primitive functions (e.g., sin, exp, step) combined using differentiable gates. This allows the network to learn which symbolic function best fits the data while remaining differentiable for training.

**Acceptance criteria:**
    - `python/carnot/models/s2kan.py` implements the primitive functions (sin, exp, step) and differentiable gating logic.
    - Tests verify the differentiable gating and primitive evaluations.
    - Test coverage for `s2kan.py` is 100%.
    - `results/experiment_1857_s2kan.json` is generated upon success.

### SCENARIO-KAN-1857: S2KAN primitives and differentiable gates

Given the `s2kan.py` module,
When the differentiable gates and primitives (sin, exp, step) are evaluated and tested,
Then the tests pass with 100% coverage, and `results/experiment_1857_s2kan.json` is written.

## REQ-KAN-1858: GloroKAN Lipschitz Bounds in S2KAN

The KAN capability MUST implement a Lipschitz approximation pass for the Softly Symbolified KANs (S2KAN).

**Rationale:**
    Robustness verification for KANs requires Lipschitz bounds (GloroKAN). Extending the KAN model forward pass to output local Lipschitz bounds enables formal property checks.

**Acceptance criteria:**
    - `python/carnot/models/s2kan.py` extends the model forward pass to output local Lipschitz bounds.
    - Tests verify the bounds hold mathematically.
    - Test coverage for the new code in `s2kan.py` is 100%.
    - `results/experiment_1858_glorokan.json` is generated upon success.

### SCENARIO-KAN-1858: S2KAN Lipschitz bounds

Given the `s2kan.py` module,
When the forward pass returns Lipschitz bounds and tests mathematically verify the bounds,
Then the tests pass with 100% coverage, and `results/experiment_1858_glorokan.json` is written.

## REQ-KAN-1859: S2KAN Z3 Verification

The KAN capability MUST connect S2KAN symbolic primitives to Z3 for formal verification.

**Rationale:**
    Verification tiers must be formally proven via Z3/MILP. A transpiler script must convert S2KAN primitives to Z3 constraints to verify consistency over a bounded input domain.

**Acceptance criteria:**
    - A script transpiles S2KAN layer into Z3 constraints.
    - Consistency over a bounded input domain is formally proven using Z3.
    - Test coverage for the new code is 100%.
    - `results/experiment_1859_z3_verify.json` is generated upon success.

### SCENARIO-KAN-1859: Z3 Transpilation and Verification

Given the `s2kan.py` primitives and a Z3 transpilation script,
When S2KAN operations are converted to Z3 constraints and verified for a bounded domain,
Then the proof completes successfully, tests pass with 100% coverage, and `results/experiment_1859_z3_verify.json` is written.

## REQ-KAN-1862: E2E Test Verifying S2KAN Constraints using Flagship MoE

The KAN capability MUST run an E2E test verifying S2KAN constraints using the mandated SOTA model (`unsloth/Qwen3.6-35B-A3B-GGUF`).

**Rationale:**
    S2KAN constraints must be verified in an end-to-end pipeline using the flagship MoE model to ensure the integration between the SOTA model and S2KAN verifier functions correctly.

**Acceptance criteria:**
    - `python/carnot/pipeline/experiment_1862.py` (or similar script) loads `unsloth/Qwen3.6-35B-A3B-GGUF`.
    - The script passes output to the S2KAN verifier to verify constraints.
    - Test coverage is 100%.
    - `results/experiment_1862_e2e.json` is written upon success containing the model ID.

### SCENARIO-KAN-1862: S2KAN constraints verified end-to-end with Qwen3.6-35B

Given the S2KAN layer and the mandated flagship MoE,
When the end-to-end test runs,
Then it loads the Qwen3.6-35B model, verifies the constraints using S2KAN, and writes the `results/experiment_1862_e2e.json` artifact.

## REQ-KAN-1926: S2KAN Symbolic Primitives Dictionary

The KAN capability MUST implement an extensible dictionary of symbolic primitives and learnable gates for S2KAN.

**Rationale:**
    To support arbitrary functional forms during symbolic discovery, S2KAN needs a dictionary of primitives (e.g. sin, exp) and learnable gates that enforce symbolic constraints.

**Acceptance criteria:**
    - Code implements a dictionary of symbolic primitives and learnable gates.
    - Code validates the model against a known functional form.
    - Tests verify the logic and achieve 100% test coverage.
    - `results/experiment_1926_s2kan_symbolic.json` is generated upon success.

### SCENARIO-KAN-1926: S2KAN Dictionary and Learnable Gates

Given the S2KAN primitives dictionary and learnable gates,
When validated against a known functional form,
Then the tests pass with 100% coverage, and `results/experiment_1926_s2kan_symbolic.json` is written.

## REQ-KAN-2005: Adaptive Energy Landscape KAN Topology Updates

The KAEM/KAN energy model tier MUST expose a deterministic adaptive mesh
refinement pass for one-dimensional energy splines.  The pass MUST measure
local landscape complexity from spline slope changes, insert knots into
under-resolved complex intervals, remove interior knots from smooth intervals,
preserve boundary knots, and keep the spline evaluable after the topology
change.  It MUST emit JSON-safe structural-change metrics including the knot
counts before and after refinement, counts of added and removed knots,
complexity thresholds, and changed interval positions.

The experiment MUST write
`results/experiment_2005_adaptive_energy_landscapes_kan.json` with `schema`,
`status`, `experiment_id`, `spec_traces`, `run_date`,
`structural_change_metrics`, `energy_probe`, `adaptive_mesh_refinement_ready`,
`tests_run`, and a terminal `honest_verdict`.

**Rationale:**
    Tier 4 learning needs the energy function's function space to change when
    the observed landscape changes.  Adding knots around high-curvature regions
    gives later fitting steps more local capacity, while removing smooth
    interior knots keeps dormant structure from growing without evidence.

**Acceptance criteria:**
    - `UnivariateKAEMLayer.adaptive_mesh_refine()` updates knot positions and
      control points deterministically from local complexity scores.
    - Complex landscapes add at least one knot when below the configured maximum.
    - Smooth landscapes remove interior knots when above the configured minimum.
    - Structural-change metrics are serializable and preserve the before/after
      topology evidence needed by the conductor.
    - `write_adaptive_energy_landscape_kan_artifact()` writes a complete Exp
      2005 JSON artifact with `REQ-KAN-2005` and `SCENARIO-KAN-2005` traces.

### SCENARIO-KAN-2005: Adaptive KAEM spline topology artifact

Given a deterministic KAEM layer containing both complex and smooth marginal
energy splines,
When adaptive mesh refinement is applied and the Exp 2005 artifact is written,
Then complex regions gain knots, smooth regions lose knots where allowed,
the refined energy remains finite on probe inputs, structural-change metrics
record the topology delta, and
`results/experiment_2005_adaptive_energy_landscapes_kan.json` is complete.

## REQ-KAN-2070: GloroKAN Robustness Verification for CarnotKAN

The KAN capability MUST implement local Lipschitz constant approximation via B-splines
to verify the robustness of `CarnotKAN`. It MUST record verification bounds to JSON.

**Rationale:**
    GloroKAN leverages algebraic geometry of B-splines to verify KAN robustness.
    By finding the local Lipschitz constant, we can ensure the network's behavior
    is bounded under small perturbations.

**Acceptance criteria:**
    - `python/carnot/models/kan/glorokan.py` implements the GloroKAN robustness verifier and `CarnotKAN`.
    - Tests verify the robustness of a small synthetic constraint system.
    - `results/experiment_2070_glorokan.json` is written with bounds and status.

### SCENARIO-KAN-2070: Verify CarnotKAN robustness

Given a trained or initialized CarnotKAN,
When GloroKAN robustness verification is applied via B-splines algebraic geometry,
Then it computes the local Lipschitz bounds, records them, and outputs to
`results/experiment_2070_glorokan.json`.

## REQ-KAN-2071: Discrete Symbolic Embedding in CarnotKAN

The KAN capability MUST add discrete symbolic embedding capabilities to `CarnotKAN`.
It MUST implement hierarchical gating for symbolic primitive discovery and be able to
train on a simple logic task (e.g., AND/XOR constraints) to measure the accuracy of
discovered symbols.

**Rationale:**
    To support interpretable symbolic primitives, `CarnotKAN` needs hierarchical
    gating that can discover correct symbolic operations from data.

**Acceptance criteria:**
    - `python/carnot/models/kan/glorokan.py` (or appropriate module) implements
      hierarchical gating in `CarnotKAN`.
    - Tests verify the logic on a simple logic task and achieve 100% test coverage.
    - `results/experiment_2071_symbolic_kan.json` is generated correctly.

### SCENARIO-KAN-2071: Symbolic primitive discovery via hierarchical gating

Given `CarnotKAN` with discrete symbolic embeddings,
When trained on a simple logic task (e.g., AND/XOR constraints),
Then it discovers the correct symbols with measurable accuracy, tests pass,
and `results/experiment_2071_symbolic_kan.json` is written.

## REQ-KAN-2060: KAN Symbolizer for exact symbolic formulas

The KAN capability MUST provide a way to extract exact symbolic formulas from trained KAN energy functions.

**Rationale:**
    KAN4CBC relies on symbolization of 1D splines to generate SMT-verifiable expressions. The symbolizer converts the piecewise representations into algebraic AST strings that can be evaluated or verified externally.

**Acceptance criteria:**
    - `python/carnot/verify/kan_symbolizer.py` implements `KANSymbolizer`.
    - It extracts knot points and polynomial coefficients from `UnivariateKAEMLayer`.
    - It generates a symbolic AST string representing the constraints.
    - Test coverage is 100% in `tests/python/verify/test_kan_symbolizer.py`.
    - Results are logged to `results/experiment_2060_kan_symbolizer.json`.

### SCENARIO-KAN-2060: Extract symbolic piecewise expressions

Given a trained `UnivariateKAEMLayer`,
When `KANSymbolizer` is used to extract polynomials and an AST string,
Then the correct piecewise linear intercepts and slopes are retrieved,
and an SMT-verifiable AST string is returned.

## REQ-KAN-2083: KAN4CBC MILP Z3 Verification

The KAN capability MUST implement formal verification properties for the MILP KAN using the Z3 SMT solver.

**Rationale:**
    Using techniques from KAN4CBC, we can use SMT solvers to verify the safety and correctness of the KAN.
    Connecting the MILP representation to the Z3 solver allows asserting robustness properties.

**Acceptance criteria:**
    - `python/carnot/models/kan/kan4cbc.py` connects a MILP representation to the `z3-solver`.
    - It asserts a simple robustness property and attempts to verify it.
    - SMT solver execution time and result are logged to `results/experiment_2083_kan4cbc.json`.
    - Test coverage is 100%.

### SCENARIO-KAN-2083: Verify KAN MILP robustness using Z3

Given a MILP KAN representation,
When a simple robustness property is asserted and verified via Z3,
Then the solver execution time and boolean result are logged to `results/experiment_2083_kan4cbc.json`.

## REQ-KAN-1781: KANelE Look-Up Table (LUT) evaluations

The KAN capability MUST implement KANelE Look-Up Table evaluations in Python.

**Rationale:**
    Transforming a small KAN tier to LUT format is essential for KANelE hardware accounting and FPGA deployment blueprints without requiring immediate hardware synthesis.

**Acceptance criteria:**
    - `scripts/experiment_1781_kan_lut.py` implements the conversion of a small KAN tier to LUT format.
    - `results/experiment_1781_kan_lut.json` is generated with `schema: "carnot.kan.lut.v1"` and `lut_conversion_success: true`.
    - Tests verify the conversion logic and achieve 100% test coverage for the new code.

### SCENARIO-KAN-1781: Transform a small KAN tier to LUT format

Given a small mock KAN tier,
When the LUT transformation logic runs,
Then it outputs the LUT evaluation formats and `results/experiment_1781_kan_lut.json` is written with `lut_conversion_success: true`.

## REQ-KAN-1782: Benchmark hardware accounting for KAN LUTs

The KAN capability MUST implement a benchmark for hardware accounting of KAN LUTs. It MUST measure Bit Operations (BOPs) and Number of Additions and Bit-Shifts (NABS) and explicitly claim no hardware execution.

**Rationale:**
    To ensure we have hardware accounting benchmarks before attempting hardware synthesis or execution.

**Acceptance criteria:**
    - `scripts/experiment_1782_kan_benchmark.py` implements the hardware accounting logic.
    - `results/experiment_1782_kan_benchmark.json` is generated with `schema: "carnot.kan.benchmark.v1"`, `bops`, `nabs`, and `hardware_execution_claim: false`.
    - Tests verify the benchmark logic and achieve 100% test coverage for the new code.

### SCENARIO-KAN-1782: Measure BOPs and NABS for KAN LUTs

Given a mock KAN tier or LUT,
When the benchmark logic runs,
Then it outputs BOPs and NABS, and `results/experiment_1782_kan_benchmark.json` is written with `hardware_execution_claim: false`.

## REQ-KAN-1803: KAEM Energy Function and Inverse Transform Sampling

The KAN capability MUST implement a Kolmogorov-Arnold Energy Model (KAEM) structure
using 1D B-splines instead of dense layers for its energy function. It MUST also
implement an inverse transform sampling method allowed by the univariate splines
to bypass MCMC.

**Rationale:**
    Replacing MLP-based energy functions with KAEM univariate splines (KART) allows
    for exact inference and interpretability without relying on MCMC sampling.

**Acceptance criteria:**
    - `python/carnot/models/kaem.py` exposes the `KAEMEnergy` model.
    - The model uses 1D B-splines for energy evaluation.
    - The model provides an `inverse_transform_sample` method that bypasses MCMC.
    - Tests verify the energy function and inverse transform sampling, referencing `REQ-KAN-1803` and `SCENARIO-KAN-1803`.
    - `results/experiment_1803_kaem_proto.json` is generated upon success.

### SCENARIO-KAN-1803: KAEM exact inference via inverse transform sampling

Given an initialized KAEM model using 1D B-splines,
When inverse transform sampling is invoked to draw samples,
Then it generates samples analytically without MCMC, tests pass with 100% coverage,
and `results/experiment_1803_kaem_proto.json` is written.

## REQ-KAN-1909: Wahkon RKHS Architecture

The KAN capability MUST implement the Wahkon architecture (arXiv:2605.14041),
which uses a Reproducing Kernel Hilbert Space (RKHS) alternative to standard
KANs to provide finite-sample guarantees.

**Rationale:**
Standard KANs lack finite-sample guarantees. Wahkon introduces an RKHS
formulation that offers bounds on finite-sample convergence while retaining
the KAN-like additive structure.

**Acceptance criteria:**
- `python/carnot/pipeline/wahkon_rkhs.py` exposes a Wahkon RKHS model.
- Tests verify the model's initialization and forward pass, referencing
  `REQ-KAN-1909` and `SCENARIO-KAN-1909`.
- Tests achieve 100% test coverage.
- `results/experiment_1909_wahkon.json` is generated upon success.

### SCENARIO-KAN-1909: Wahkon RKHS Model Initialization and Forward Pass

Given a Wahkon RKHS model,
When initialized and a forward pass is executed,
Then it computes the correct output shape and `results/experiment_1909_wahkon.json` is written.

## REQ-KAN-2034: KAGNN Verifier for Graph Coloring

The KAN capability MUST implement a Kolmogorov-Arnold Graph Neural Network (KAGNN) verifier for Graph Coloring constraints.

**Rationale:**
    Graph constraints like Graph Coloring require relational verification between adjacent nodes. Standard linear GNNs are opaque. By using a Symbolic-KAN routing layer instead of linear weights on the edges, we can explicitly learn and extract the constraint (e.g., equality or inequality) driving the graph coloring verification.

**Acceptance criteria:**
    - `python/carnot/models/ising/kagnn.py` exposes a `KAGNNLayer` or `KAGNNVerifier` that uses Symbolic-KAN routing logic to evaluate edge constraints.
    - Tests verify that it correctly assigns lower energy to valid graph colorings and higher energy to invalid ones.
    - Test coverage for the new module is 100%.
    - `results/experiment_2034_kagnn.json` is generated upon success.

### SCENARIO-KAN-2034: KAGNN Verifier Evaluates Graph Coloring

Given a KAGNN verifier using Symbolic-KAN splines,
When evaluated on valid and invalid graph coloring instances,
Then it outputs lower energy for valid colorings and higher energy for invalid ones, tests pass with 100% coverage, and `results/experiment_2034_kagnn.json` is written.



## REQ-KAN-2035: Comparative Evaluation of KAGNN vs MLP

The KAN capability MUST evaluate the tangible efficiency or accuracy benefit of KAGNN against an MLP baseline.

**Rationale:**
    To validate that KAGNN provides a tangible efficiency or accuracy benefit over a standard MLP for constraint verification on graph coloring.

**Acceptance criteria:**
    - `scripts/eval_kagnn_vs_mlp.py` generates a synthetic dataset of constraint graphs and runs a comparative evaluation between KAGNN and an MLP baseline.
    - Tests verify the evaluation script or the underlying logic.
    - `results/exp2035_kagnn_eval.json` is generated upon success containing the evaluation results.

### SCENARIO-KAN-2035: KAGNN vs MLP Comparative Evaluation

Given a synthetic dataset of constraint graphs,
When the comparative evaluation is run,
Then it outputs the results comparing KAGNN and MLP, and `results/exp2035_kagnn_eval.json` is written.


## REQ-KAN-2104: Alignment-Symmetry Quantization for KAN Tiers

The KAN capability MUST implement ASP-KAN-HAQ (arXiv:2509.07xxx) grid-alignment and symmetry sharing for B-splines.

**Rationale:**
    Aligning spline knots with quantization grids and using symmetry sharing massively reduces KAN hardware area.

**Acceptance criteria:**
    - `python/carnot/hardware/asp_kan_quant.py` implements the ASP-KAN-HAQ alignment and symmetry sharing algorithm for B-splines.
    - Tests verify that the quantized forward pass matches the expected logic compared to the full FP32 pass.
    - `results/experiment_2104_asp_kan.json` is written with `asp_kan_ready=true`.
    - Tests pass with 100% coverage.

### SCENARIO-KAN-2104: ASP-KAN-HAQ Grid Alignment

Given a set of B-spline knots and control points,
When ASP-KAN-HAQ grid alignment and symmetry sharing are applied,
Then the knots align to the quantization grid, symmetry is enforced, and the quantized forward pass is evaluated, successfully logging to `results/experiment_2104_asp_kan.json`.

## REQ-KAN-3374: EBT Sidecar Scoring with KAN Energy

The sidecar scoring pipeline MUST support a KAN energy formulation that computes
an additional energy term for candidate scoring.

### SCENARIO-KAN-3374: Integrate KAN energy formulation

Given a sidecar record with KAN energy features,
When the sidecar replay scorer evaluates the candidate,
Then the KAN energy formulation is correctly integrated, and the test succeeds, logging to `results/experiment_3374_ebt_kan_integration.json`.
