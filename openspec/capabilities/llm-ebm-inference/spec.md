# LLM-EBM Inference Capability Specification

**Capability:** llm-ebm-inference
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11, FR-12

## Overview

Defines the first concrete "anti-hallucination" pipeline: an LLM proposes a solution to a constraint satisfaction problem (SAT, graph coloring), and the EBM verifies it, repairs violations via gradient descent, and issues a verification certificate. This bridges Carnot's verifiable reasoning (FR-12) with LLM output, demonstrating that energy-based verification can catch and fix hallucinated reasoning.

The pipeline reuses existing primitives (BaseConstraint, ComposedEnergy, repair(), VerificationResult) and adds domain-specific constraint encoders plus an LLM output parser.

## Requirements

### REQ-INFER-001: SAT Constraint Encoding

The system shall encode CNF SAT clauses as differentiable energy terms:
- Variables x_i represented as continuous values in [0, 1] (0 = False, 1 = True)
- Each clause uses product relaxation: E = prod(1 - literal_value) for each literal
  - Energy = 0 when any literal is true (clause satisfied)
  - Energy = 1 when all literals are false (clause violated)
- A binary penalty constraint pushes variables toward {0, 1}: E = sum(x_i * (1 - x_i))
- The system shall support DIMACS CNF format parsing
- `build_sat_energy()` shall produce a ComposedEnergy with one constraint per clause plus binary penalty

### REQ-INFER-002: Graph Coloring Constraint Encoding

The system shall encode graph coloring as differentiable energy terms:
- Each node's color is a continuous value in [0, n_colors - 1]
- Adjacent nodes must have different colors: E = max(0, 1 - |x_a - x_b|)^2
- A range penalty keeps values in bounds: E = sum(max(0, -x_i)^2 + max(0, x_i - (n_colors-1))^2)
- `build_coloring_energy()` shall produce a ComposedEnergy with one constraint per edge plus range penalty

### REQ-INFER-003: LLM Output Parsing

The system shall parse LLM text output into JAX arrays for verification:
- SAT: supports "1 0 1 0", "x1=True, x2=False", "T F T F" formats
- Graph coloring: supports "0 1 2 0 1" and "node 0: red, node 1: blue" formats
- Raises ValueError on unparseable input or wrong variable count

### REQ-INFER-004: Verify-and-Repair Pipeline

The system shall provide a complete verify-and-repair pipeline:
1. Score the LLM's assignment against the energy function
2. If constraints are violated, run gradient repair (reusing existing `repair()`)
3. Round the repaired continuous assignment to discrete values
4. Re-verify the rounded result
5. Return a VerifyRepairResult containing initial, repaired, and rounded verifications plus the repair trajectory

### REQ-INFER-005: Benchmark Harness

The system shall provide a benchmark harness that:
- Generates random SAT instances (configurable n_vars, n_clauses, clause_size)
- Generates random graph coloring instances (configurable n_nodes, n_colors, edge probability)
- Generates random assignments (simulating LLM output)
- Runs the verify-and-repair pipeline on each instance
- Reports aggregated statistics (initial violations, repaired violations, success rate, energy reduction)

## Scenarios

### SCENARIO-INFER-001: SAT Clause Verification

**Given** a 3-SAT clause (x1 OR NOT x2 OR x3)
**And** assignment x1=True, x2=True, x3=False
**When** the clause constraint energy is computed
**Then** energy is approximately 0 (clause satisfied by x1=True)

**Given** the same clause with assignment x1=False, x2=False, x3=False
**When** energy is computed
**Then** energy is approximately 1 (clause violated: x1=F, NOT x2=T but x3=F... wait, NOT x2=True so clause IS satisfied)
**Corrected**: assignment x1=False, x2=True, x3=False → NOT x2=False, all literals false → energy ≈ 1

### SCENARIO-INFER-002: DIMACS Parsing

**Given** a DIMACS CNF string with header "p cnf 3 2" and clauses "1 -2 3 0" and "-1 2 0"
**When** parsed
**Then** produces 2 SATClause objects with correct variable indices and negation flags
**And** n_vars = 3

### SCENARIO-INFER-003: Graph Coloring Verification

**Given** a triangle graph (3 nodes, all pairs connected)
**And** coloring [0, 1, 2] (all different)
**When** coloring energy is computed
**Then** energy is approximately 0 (no adjacent same-color)

**Given** coloring [0, 0, 1] (nodes 0 and 1 share color)
**When** energy is computed
**Then** energy > 0 (edge 0-1 violated)

### SCENARIO-INFER-004: Gradient Repair Reduces Violations

**Given** a random 3-SAT instance with 20 variables and 60 clauses
**And** a random assignment violating multiple clauses
**When** gradient repair runs for up to 200 steps
**Then** the number of violated clauses decreases
**And** total energy decreases monotonically (or at least overall)

### SCENARIO-INFER-005: Full Pipeline

**Given** a SAT instance with a known satisfying assignment
**And** a deliberately wrong assignment (simulating LLM hallucination)
**When** verify_and_repair() is called
**Then** initial_verification shows violations
**And** repaired_verification shows fewer violations
**And** the result contains a repair trajectory

### SCENARIO-INFER-006: Binary Penalty

**Given** SAT variables at x_i = 0.5 (maximally ambiguous)
**When** binary penalty energy is computed
**Then** energy is maximal
**Given** variables at x_i ∈ {0, 1}
**Then** binary penalty energy is 0

### REQ-INFER-006: LLM Solver Integration

The system shall provide functions to send constraint problems to an LLM and parse the responses:
- `solve_sat_with_llm()`: builds SAT prompt, calls OpenAI-compatible API, returns raw text
- `solve_coloring_with_llm()`: builds coloring prompt, calls API
- `run_llm_sat_experiment()`: full pipeline — LLM solve → parse → verify → repair → certify
- `run_llm_coloring_experiment()`: full pipeline for coloring
- Graceful degradation when openai package is not installed or API fails

### SCENARIO-INFER-007: Full LLM Pipeline

**Given** a SAT instance sent to an LLM via the solver
**When** the LLM returns a (possibly incorrect) assignment
**Then** the assignment is parsed and verified against the energy function
**And** if violated, gradient repair improves the assignment
**And** the full VerifyRepairResult contains initial, repaired, and rounded verifications

### REQ-INFER-007: Learned Energy Functions

The system shall support training EBMs to learn verification criteria from (correct, incorrect) example pairs:
- Generate training data via rejection sampling (satisfying assignments = data, random = noise)
- Train using NCE loss from `carnot.training.nce`
- Wrap the trained model as a `BaseConstraint` for use in the verify-and-repair pipeline
- Provide comparison tools to measure learned verifier accuracy against hand-coded ground truth

### SCENARIO-INFER-008: Learned SAT Verifier

**Given** a random 3-SAT instance with 5 variables and 15 clauses
**And** a Gibbs model trained on satisfying vs random assignments via NCE
**When** the trained model scores a new satisfying assignment
**Then** its energy is lower than for a violating assignment
**And** the learned verifier classifies at least 70% of test assignments correctly
**And** verify-and-repair using the learned energy improves random assignments

### REQ-INFER-014: Hallucination Direction Detection

The system shall detect hallucination by finding the principal direction in activation space that separates correct from hallucinated LLM outputs:
- Compute mean-difference direction between correct and hallucinated activation clusters
- Optional SVD refinement for top-k distinguishing directions
- Hallucination energy: projection of activation onto the direction (high = likely hallucination)
- HallucinationDirectionConstraint wraps the energy for ComposedEnergy integration
- One-sided (ReLU) penalty: only penalizes projections toward hallucination, not away

### SCENARIO-INFER-014-001: Hallucination Direction Discovery

**Given** per-layer activations from 50 correct and 50 hallucinated outputs
**When** `find_hallucination_direction()` is called
**Then** the discovered direction aligns (cosine similarity > 0.8) with the true separation axis
**And** hallucinated activations get higher energy than correct ones
**And** the direction is unit-normalized by default

### REQ-INFER-015: EBM-Guided Rejection Sampling

The system shall combine per-token activation energy (from a trained Gibbs EBM) with logprob scores for candidate selection:
- Generate N candidates with sampling
- Extract per-token hidden states from a configurable transformer layer
- Score each token through the trained EBM (low energy = likely correct)
- Combine as composite: ebm_weight * mean_ebm_energy - logprob_weight * mean_logprob
- Select candidate with lowest composite energy
- Configurable weights allow pure EBM, pure logprob, or weighted combination

### SCENARIO-INFER-015-001: EBM Scoring Correctness

**Given** a Gibbs EBM trained on correct vs wrong activations
**And** per-token activations from a generated response
**When** `score_activations_with_ebm()` is called
**Then** the mean energy is finite and reflects the EBM's learned distinction
**And** activations from the trained distribution get lower energy than out-of-distribution

### REQ-INFER-016: Multi-Layer Hallucination Probing

The system shall probe each transformer layer independently to find where hallucination signal is strongest:
- Train a small Gibbs EBM probe at each layer using NCE loss
- Report per-layer train/test accuracy and energy gap
- Identify the layer with highest test accuracy as the best probe layer
- Support probing a subset of layers for efficiency

### SCENARIO-INFER-016-001: Best Layer Identification

**Given** per-layer activations from correct and wrong model outputs with varying separability
**When** `probe_all_layers()` is called
**Then** the layer with highest separation is identified as best_layer
**And** the best layer's test accuracy exceeds layers with lower separation
**And** results include per-layer accuracy, gap, and train/test counts

### REQ-INFER-017: Diffusion-of-Thought Text Refinement

The system SHALL expose `DiffusionOfThought` as an inference-time refinement
mode over an existing verifier ensemble energy.  The refinement SHALL split a
response into whitespace tokens, estimate each token's violation contribution
by comparing the original response energy against the energy after masking that
token, propose deterministic replacement candidates when no inpainter LLM is
available, and iteratively select the replacement that minimises composite
energy.

The refinement SHALL support timestep counts `T in {1, 5, 25, 125}` for
compute/accuracy trade-off reporting.  Each call SHALL return the refined text
and an energy trace that includes the initial energy and one entry per
refinement step.

### SCENARIO-INFER-017-001: DoT Masks And Repairs High-Energy Tokens

**Given** a verifier ensemble that assigns higher energy to an arithmetic
violation token
**When** `DiffusionOfThought.refine()` runs for one or more steps
**Then** the highest-energy token is selected for candidate replacement
**And** the returned energy trace is non-increasing
**And** the refined response has no higher composite energy than the baseline

### REQ-INFER-SOTA-004: SOTA GGUF Cache Provenance Preflight

The system SHALL provide a local-only cache/provenance preflight artifact before
mandated SOTA GGUF headline inference.  The preflight SHALL inspect the three
mandated model IDs `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, record each model's cached status,
`model_path`, expected quantization, and observed quantization suffix, and call
`cached_sota_pair(gpu_indices=(0, 1))` only through the existing safe local
cache resolver.

The artifact SHALL expose `status`, `cached_sota_ready`,
`headline_result_possible`, `models_used`, `missing_models`,
`model_specs_preview`, `provenance_ok`, and `honest_verdict`.  It SHALL set
`cached_sota_ready=true` only when at least one mandated SOTA GGUF is cached and
`cached_sota_pair()` returns two loadable `MODEL_SPECS` entries with
`model_path` values.  Otherwise it SHALL set `cached_sota_ready=false`, record
`missing_models`, carry Exp 1296 prior-failure coverage into the artifact, and
emit an honest blocked/not-ready verdict without downloading models.

### SCENARIO-INFER-SOTA-004-001: Cached Pair Allows Headline Work

**Given** local cache hits for the mandated Qwen and Gemma 26B GGUF models
**And** Exp 1296 prior-failure coverage passed
**When** the SOTA GGUF preflight runs
**Then** `cached_sota_ready` is true
**And** `headline_result_possible` is true
**And** `models_used` names the two `cached_sota_pair()` entries
**And** every model preview row records cache and quantization provenance.

### SCENARIO-INFER-SOTA-004-002: Missing Cache Blocks Headline Work

**Given** no mandated SOTA GGUF file can be resolved from local cache
**When** the SOTA GGUF preflight runs
**Then** `cached_sota_ready` is false
**And** `headline_result_possible` is false
**And** `missing_models` names the uncached mandated model IDs
**And** `honest_verdict` does not claim a headline result is possible.

### SCENARIO-INFER-SOTA-004-003: Prior-Failure Coverage Gates Provenance

**Given** Exp 1296 prior-failure coverage is missing or failed
**And** the local cache resolver can still inspect GGUF files
**When** the SOTA GGUF preflight runs
**Then** `provenance_ok` is false
**And** `headline_result_possible` is false
**And** the artifact preserves the Exp 1296 coverage fields that caused the
provenance block.

### REQ-INFER-SOTA-005: Cached Pair Resolver Uses Any Two Mandated Local GGUFs

`cached_sota_pair()` SHALL inspect the three mandated SOTA GGUF model IDs through
the safe local resolver and SHALL return exactly two loadable `MODEL_SPECS`
entries when at least two mandated models are cached locally.  The pair SHALL be
deterministic: cached models are selected in registry order and assigned to the
provided `gpu_indices` order.  Missing third mandated models are optional for
pair readiness and SHALL be recorded by resolver-repair artifacts as
`missing_optional_models`; the helper SHALL NOT download models while resolving
the pair.

If fewer than two mandated SOTA GGUFs are cached locally, `cached_sota_pair()`
SHALL return `None` so callers can block or fall back honestly instead of
attempting a partial headline pair.

### SCENARIO-INFER-SOTA-005-001: Two Cached Mandated Models Return Pair

**Given** local cache hits for two mandated SOTA GGUF models
**And** a cache miss for the remaining mandated model
**When** `cached_sota_pair(gpu_indices=(0, 1))` is called
**Then** it returns two `MODEL_SPECS` entries with `model_path` values
**And** the missing third model is treated as optional for pair readiness.

### SCENARIO-INFER-SOTA-005-002: One Cached Mandated Model Blocks Pair

**Given** exactly one mandated SOTA GGUF model resolves from local cache
**When** `cached_sota_pair()` is called
**Then** it returns `None`.

### SCENARIO-INFER-SOTA-005-003: No Cached Mandated Models Block Pair

**Given** no mandated SOTA GGUF model resolves from local cache
**When** `cached_sota_pair()` is called
**Then** it returns `None`.

### REQ-INFER-SOTA-006: Cached SOTA GGUF llama.cpp Smoke-Load Probe

The system SHALL provide a minimal local smoke-load artifact for the cached SOTA
GGUF pair before downstream certificate runs use those models.  The probe SHALL
write `results/experiment_1310_sota_gguf_llamacpp_smoke_load.json` with
`status="in_progress"` before live checks, SHALL read Exp 1309, and SHALL abort
to an honest terminal blocker when `sota_pair_ready` is not true.

When Exp 1309 is ready, the probe SHALL resolve the pair via
`cached_sota_pair(gpu_indices=(0, 1))` and record each resolved model's exact
`hf_id`, `name`, `model_path`, observed quantization suffix, and GPU assignment.
The model IDs SHALL be drawn only from the mandated headline set
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`.

The probe SHALL import the project-local llama.cpp runtime (`from llama_cpp
import Llama`) before generation.  If the import fails, it SHALL record
`llama_cpp_import_ok=false` with the exact exception string and SHALL NOT fake
generation.  If the import succeeds, the probe SHALL attempt one deterministic,
tiny-token generation per resolved model and record per-model load success,
token count, elapsed seconds, tokens per second, and GPU memory when available.

The final artifact SHALL expose `status`, `models_loaded`,
`llama_cpp_import_ok`, `tokens_per_second`, `gpu_memory_gb`,
`model_specs_count`, `models_used`, `headline_result_possible`, and
`honest_verdict`.  `headline_result_possible` SHALL be true only when two
mandated headline GGUFs were resolved and both smoke generations completed with
at least one emitted token.

### SCENARIO-INFER-SOTA-006-001: Ready Pair Smoke-Loads

**Given** Exp 1309 reports `sota_pair_ready=true`
**And** `cached_sota_pair(gpu_indices=(0, 1))` returns two mandated GGUF specs
**And** the llama.cpp import succeeds
**When** the smoke-load probe runs with a tiny deterministic prompt
**Then** the artifact records both model identities, paths, quantization suffixes,
GPU assignments, per-model throughput, aggregate throughput, and
`headline_result_possible=true`.

### SCENARIO-INFER-SOTA-006-002: Missing llama.cpp Blocks Without Fake Generation

**Given** Exp 1309 reports `sota_pair_ready=true`
**And** the cached pair resolves
**When** importing `llama_cpp.Llama` raises an exception
**Then** the artifact records `llama_cpp_import_ok=false`
**And** every resolved model remains ungenerated
**And** `headline_result_possible=false`.

### SCENARIO-INFER-SOTA-006-003: Exp 1309 Not Ready Blocks Probe

**Given** Exp 1309 does not report `sota_pair_ready=true`
**When** the smoke-load probe runs
**Then** it writes a complete terminal blocker artifact
**And** it does not import llama.cpp or attempt generation.

### REQ-INFER-SOTA-007: Live SOTA Repair Runtime Gate

The system SHALL provide a fail-fast local runtime gate for the mandated SOTA
GGUF repair models before downstream repair experiments report headline claims.
The gate SHALL inspect only local cache state for
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, SHALL probe GPU availability, and SHALL
attempt one tiny repair-style live inference request on at least one mandated
model that is locally available.

The artifact SHALL expose `status`, `model_specs`, `local_sota_runtime_ready`,
`live_sota_model_inference_used`, `models_found_in_cache`,
`models_missing_from_cache`, `gpu_probe`, `smoke_inference_results`,
`blockers`, and `honest_verdict`.  Each smoke result SHALL record the exact
command, runtime mode, stdout and stderr summaries, and whether the response was
truly live.  `local_sota_runtime_ready` SHALL be true only when at least one
mandated local SOTA GGUF model completes live inference and emits a usable
response; otherwise the artifact SHALL remain an honest terminal blocker without
downloading model files.

### SCENARIO-INFER-SOTA-007-001: Live Local Inference Opens Gate

**Given** at least one mandated SOTA GGUF model is present in local cache
**And** the llama.cpp runtime can load it
**When** the live SOTA repair runtime preflight runs
**Then** the artifact records the model as found
**And** records a smoke inference result with `truly_live=true`
**And** sets `local_sota_runtime_ready=true` and
`live_sota_model_inference_used=true`.

### SCENARIO-INFER-SOTA-007-002: Missing Or Failed Runtime Blocks Gate

**Given** no mandated SOTA GGUF model can complete live inference locally
**When** the live SOTA repair runtime preflight runs
**Then** the artifact records missing models or runtime failures in `blockers`
**And** sets `local_sota_runtime_ready=false`
**And** does not claim live SOTA model inference was used.

### REQ-INFER-SOTA-008: Local SOTA GGUF Runtime Repair Artifact

The system SHALL provide an Exp 1463 runtime-repair artifact for the mandated
local SOTA GGUF path.  The repair SHALL write
`results/experiment_1463_local_sota_gguf_runtime_repair.json` with
`status="in_progress"` before runtime probes, SHALL reproduce the prior Exp
1442 runtime gate verdict before applying any path repair, SHALL inspect CUDA
runtime discovery evidence from `ldconfig`, `nvidia-smi`, Python package
metadata, environment variables, and local library files, and SHALL attempt the
minimum safe llama.cpp runtime-library path repair when the project-local venv
contains CUDA runtime libraries that the dynamic loader cannot currently see.

The artifact SHALL expose `status`, `model_specs`, `gpu_probe`,
`libcudart_resolution_attempted`, `missing_cache_resolution_attempted`,
`models_found_in_cache`, `models_missing_from_cache`,
`smoke_inference_results`, `live_sota_model_inference_used`,
`local_sota_runtime_ready`, `persistent_blockers`, and `honest_verdict`.
`live_sota_model_inference_used` and `local_sota_runtime_ready` SHALL be true
only when at least one mandated SOTA GGUF model produces a non-empty response
from the local llama.cpp runtime.  Otherwise the artifact SHALL remain a
complete terminal blocker with exact persistent blockers and SHALL NOT treat
legacy small models as headline evidence.

### SCENARIO-INFER-SOTA-008-001: Runtime Path Repair Enables Live SOTA

**Given** at least one mandated SOTA GGUF model is present in local cache
**And** the reproduced Exp 1442 probe failed because `libcudart.so.12` was not
visible to the loader
**When** Exp 1463 discovers project-local CUDA runtime libraries and reruns the
probe with the repaired library path
**Then** the artifact records the resolution attempt
**And** records a smoke inference result with `truly_live=true`
**And** sets `local_sota_runtime_ready=true` and
`live_sota_model_inference_used=true`.

### SCENARIO-INFER-SOTA-008-002: Persistent Blockers Stay Terminal

**Given** the cache, CUDA runtime, or llama.cpp path cannot produce any usable
mandated SOTA response
**When** Exp 1463 completes
**Then** the artifact records the exact cache or runtime blockers in
`persistent_blockers`
**And** sets `local_sota_runtime_ready=false`
**And** does not claim live SOTA model inference was used.

### REQ-INFER-SOTA-009: Live SOTA Logprob Telemetry Preflight

The system SHALL provide an Exp 1468 local telemetry preflight that verifies
whether the repaired local SOTA GGUF runtime can expose generation telemetry for
downstream HALT, Spilled Energy, and BEAVER-lite bound features.  The preflight
SHALL write `results/experiment_1468_live_sota_logprob_telemetry_preflight.json`
with `status="in_progress"` before runtime work, SHALL resolve headline models
through `cached_sota_pair()` or the Exp 1463 mandated GGUF paths, and SHALL NOT
use legacy small models as headline telemetry evidence.

The preflight SHALL request 10-20 bounded FoVer/GSM8K-style prompts from at
least one mandated SOTA GGUF model and write one JSONL row per requested case to
`results/live_sota_telemetry_manifest_1468.jsonl`.  Each row SHALL record model
identity, prompt identity, prompt text, response text, token text when exposed,
token logprob availability, top-k alternative availability, logits
availability, generation source, and exact blocker text for failures.

The artifact SHALL expose `status`, `model_specs`,
`live_sota_model_inference_used`, `telemetry_cases_requested`,
`telemetry_cases_completed`, `topk_logprobs_available`, `logits_available`,
`telemetry_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
`honest_verdict`.  `topk_logprobs_available` SHALL be true only when enough
per-token top-k alternatives exist for downstream HALT/Spilled Energy features;
runtime failures or missing llama.cpp telemetry SHALL be recorded as blockers
without running repair experiments.

### SCENARIO-INFER-SOTA-009-001: Live Top-K Telemetry Opens Downstream Path

**Given** `cached_sota_pair()` resolves at least one mandated SOTA GGUF model
**And** the llama.cpp runtime returns non-empty text with token logprobs and
top-k alternatives for bounded FoVer/GSM8K-style prompts
**When** Exp 1468 runs
**Then** the artifact records live SOTA inference as used
**And** `telemetry_cases_completed` counts the completed live generations
**And** `topk_logprobs_available=true`
**And** the JSONL manifest contains one row per generated case with response
text and per-token top-k evidence.

### SCENARIO-INFER-SOTA-009-002: Missing Runtime Telemetry Blocks Honestly

**Given** a mandated SOTA GGUF model can generate text but llama.cpp does not
return enough token logprobs, top-k alternatives, or logits
**When** Exp 1468 runs
**Then** the artifact still records the live response text and completed cases
**And** sets unavailable telemetry booleans to false
**And** preserves precise blockers for missing telemetry without claiming
headline top-k readiness.

### REQ-INFER-SOTA-010: Balanced Live SOTA Telemetry V2 Manifest

The system SHALL provide an Exp 1480 balanced telemetry manifest builder that
turns the Exp 1468 live SOTA telemetry path into an adversarially balanced
dataset for downstream validity audits.  The experiment SHALL write
`results/experiment_1480_live_sota_balanced_telemetry_v2.json` with
`status="in_progress"` before any model runtime or manifest work begins, SHALL
resolve headline models through `cached_sota_pair()` or equivalent mandated
GGUF cache paths, and SHALL NOT use legacy small models for headline telemetry
rows.

The builder SHALL request 30-40 bounded arithmetic/FoVer-style cases with known
verifier labels and a deliberate mix of correct, incorrect, format-valid, and
format-invalid expected outcomes.  It SHALL write one JSONL row per case to
`results/live_sota_balanced_telemetry_manifest_1480.jsonl`.  Each row SHALL
preserve prompt identity, prompt family, expected answer, live model identity,
response text, correctness label, format-validity label, token text when
available, token logprobs when available, top-k alternatives when available,
logits availability, exact blockers, and the generation source.

Before any telemetry diagnostic is scored, each row SHALL record superficial
baseline fields: response length, token count, JSON/schema validity, prompt
family, answer lexical overlap, and model family.  The terminal artifact SHALL
expose `status`, `model_specs`, `live_sota_model_inference_used`,
`telemetry_cases_requested`, `telemetry_cases_completed`,
`balanced_label_counts`, `topk_logprobs_available`, `logits_available`,
`superficial_baselines_recorded`, `telemetry_manifest_path`, `models_used`,
`gpu_probe`, `blockers`, `claim_allowed`, and `honest_verdict`.
`claim_allowed` SHALL default to false and SHALL become true only when the
manifest contains balanced labels, at least one real mandated local SOTA model
row, and enough telemetry for downstream adversarial audits.

### SCENARIO-INFER-SOTA-010-001: Balanced Live Rows Allow Downstream Audit

**Given** `cached_sota_pair()` resolves at least one mandated SOTA GGUF model
**And** the runtime or injected live generation path returns non-empty response
text with token telemetry for a 30-40 case balanced set
**When** Exp 1480 runs
**Then** the artifact records live SOTA inference as used
**And** `balanced_label_counts` reports both correctness labels and both format
validity labels
**And** every manifest row records the required superficial baselines before
claim gating
**And** `claim_allowed=true` only when top-k/logit evidence is available enough
for downstream adversarial audits.

### SCENARIO-INFER-SOTA-010-002: Missing Balance Or Telemetry Blocks Honestly

**Given** a mandated local SOTA model can generate text but the manifest is not
label-balanced or the runtime cannot expose enough token telemetry
**When** Exp 1480 completes
**Then** the JSONL manifest still records every completed row and all
superficial baselines
**And** unavailable telemetry booleans or balance gaps are recorded as explicit
blockers
**And** `claim_allowed=false` without falling back to legacy small model
headline rows.

### REQ-INFER-SOTA-011: Exp 1891 SOTA GGUF Cache And Runtime Preflight

The system SHALL provide an Exp 1891 local cache/runtime preflight that separates
mandated SOTA GGUF cache readiness from runtime smoke readiness before any
downstream evaluation spends a synthesis or generation call.  The preflight
SHALL define `MODEL_SPECS` for `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; SHALL inspect local cache state through the
same safe resolver/cached-pair pattern used by `cached_sota_pair()`; SHALL
attempt bounded materialization only when an explicit safe local configuration
enables it; and SHALL NOT substitute legacy small models as evidence of
headline readiness.

The artifact SHALL be written to
`results/experiment_1891_sota_gguf_cache_runtime_preflight.json` and SHALL
expose `status`, `honest_verdict`, `MODEL_SPECS`, `cache_all_available`,
`cache_any_available`, `missing_models`, `runtime_smoke_ready`,
`runtime_backend`, `models_used`, `gpu_telemetry_available`, `model_count`,
`parallel_model_count`, `fallback_runtime_candidates`, and `tests_run`.
Fallback runtimes such as llama.cpp with llguidance, vLLM.rs, and llgtrt SHALL
be recorded as candidates only unless a mandated cached GGUF completes the
runtime smoke.

### SCENARIO-INFER-SOTA-011-001: Cache And Runtime Readiness Are Separated

**Given** only a subset of mandated SOTA GGUF models is present in the local
cache
**When** Exp 1891 runs
**Then** `cache_any_available` reflects the cache hit, `cache_all_available`
remains false, and `missing_models` lists every absent mandated model
**And** runtime smoke readiness is true only when an available mandated model
completes a bounded local smoke through the selected runtime backend
**And** fallback runtime candidates are not counted as fulfilled mandated-model
evidence.

### REQ-INFER-SOTA-012: Exp 2836 SOTA Runtime And Cache Manifest

The system SHALL provide a fail-fast Exp 2836 runtime/cache manifest for the
mandated SOTA GGUF path before downstream corpus evaluations run.  The manifest
SHALL be written to `results/experiment_2836_sota_runtime_preflight.json` and
SHALL NOT run corpus evaluation.  It SHALL measure `.venv/bin/python` torch/CUDA
availability and system `python3` torch/CUDA availability separately; record
disk space, GPU memory, HuggingFace cache location, local `models/` location,
and local llama.cpp loader availability; import `scripts/experiment_template.py`
when safe and exercise `cached_sota_pair()`; inspect the mandated primary model
IDs `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; and record legacy CPU smoke-only model IDs
only as non-headline context.

For every locally available mandated SOTA GGUF selected for inspection, the
manifest SHALL record the exact path, resolved path, size in bytes, SHA256,
model family, expected quantization, observed quantization, and load-smoke
result.  If no mandated model is cached, the manifest SHALL make only a
controlled metadata-only or resumable cache attempt when local HuggingFace
tooling and credentials are already configured; otherwise it SHALL write a
`blocked_model_cache` verdict without starting a blind large download.

The artifact SHALL expose `honest_verdict`, `sota_runtime_ready`,
`selected_python`, `venv_torch_cuda_available`,
`system_python_torch_cuda_available`, `sota_models_cached`,
`cached_sota_pair_result`, `model_specs`, `preconditions_checked`, and
`duration_s`.  `sota_runtime_ready` SHALL be true only when the selected
`.venv/bin/python` reports CUDA-capable torch and at least one mandated primary
SOTA GGUF completes the bounded loader smoke as a headline model.

### SCENARIO-INFER-SOTA-012-001: Cached Mandated Model Opens Runtime Gate

**Given** `.venv/bin/python` reports `torch.cuda.is_available() == true`
**And** at least one mandated primary SOTA GGUF exists in local cache
**And** the bounded loader smoke succeeds for that file
**When** Exp 2836 builds the runtime/cache manifest
**Then** `sota_runtime_ready` is true
**And** `honest_verdict` starts with `success:`
**And** the manifest records the selected Python, model cache provenance,
`cached_sota_pair()` result, and all required precondition checks.

### SCENARIO-INFER-SOTA-012-002: Missing Cache Blocks Without Blind Download

**Given** `.venv/bin/python` reports CUDA-capable torch
**And** no mandated primary SOTA GGUF exists in local cache
**And** local HuggingFace credentials are not configured
**When** Exp 2836 builds the runtime/cache manifest
**Then** `sota_runtime_ready` is false
**And** `honest_verdict` starts with `blocked_model_cache`
**And** the manifest records the skipped cache-fill reason without attempting a
large blind download.

### SCENARIO-INFER-SOTA-012-003: System Python Mismatch Is Preserved

**Given** `.venv/bin/python` reports CUDA-capable torch
**And** system `python3` cannot import torch or reports no CUDA
**When** Exp 2836 builds the runtime/cache manifest
**Then** `venv_torch_cuda_available` and
`system_python_torch_cuda_available` preserve the mismatch as separate
booleans
**And** the manifest still gates readiness on the selected `.venv/bin/python`
rather than system `python3`.

### REQ-INFER-SOTA-013: Exp 2862 SOTA Runtime Cache/Offload Resolver V3

The system SHALL provide an Exp 2862 replacement runtime resolver artifact that
gates downstream live-model work through `sota_runtime_ready_v3`.  The artifact
SHALL be written to
`results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json`; SHALL
measure `.venv/bin/python` torch/CUDA availability, NVIDIA GPU inventory,
`llama_cpp` import/version/origin, `llama_cpp.llama_cpp.llama_supports_gpu_offload()`,
and `cached_sota_pair(gpu_indices=(0, 1))`; SHALL inspect both the local
HuggingFace cache and project `models/` layout for the mandated GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; and SHALL NOT modify
`scripts/research_conductor.py`.

The artifact SHALL expose `honest_verdict`, `sota_runtime_ready_v3`,
`model_specs`, `selected_model_hf_id`, `selected_model_path`,
`cached_sota_pair_returned_two_loadable_specs`,
`llama_cpp_gpu_offload_verified`, `preconditions_checked`,
`usable_response_count`, `total_tokens_generated`, `tokens_per_second`,
`legacy_small_models_used_only_for_smoke`, `random_seed`,
`reproducibility_checksum`, `tests_run`, `field_principles`, `run_date`, and
`duration_s`.  `sota_runtime_ready_v3` SHALL be true only when the selected
`.venv/bin/python` sees CUDA, `llama_cpp` reports GPU offload support, and at
least one mandated local SOTA GGUF emits a non-empty bounded prompt response
with one or more generated tokens.  Legacy small models MAY be named only as
CPU-smoke context and SHALL never set `sota_runtime_ready_v3=true`.

When `llama_cpp` imports but lacks GPU offload support, the artifact SHALL keep
`llama_cpp_gpu_offload_verified=false`, SHALL NOT fake readiness, and SHALL
record the exact reinstall command/environment needed to rebuild
`llama-cpp-python` with CUDA support.  When fewer than two mandated models are
cached, `cached_sota_pair_returned_two_loadable_specs` SHALL remain false even
if a single mandated model is sufficient for the v3 runtime gate.

### SCENARIO-INFER-SOTA-013-001: Single Mandated GPU Response Opens V3 Gate

**Given** `.venv/bin/python` reports CUDA-capable torch
**And** `llama_cpp` reports GPU offload support
**And** at least one mandated SOTA GGUF exists locally
**When** Exp 2862 runs one bounded prompt on that GGUF
**Then** `sota_runtime_ready_v3` is true only if the response is non-empty and
the artifact records the selected model path, generated token count,
tokens-per-second, GPU memory evidence, and real wall-clock duration.

### SCENARIO-INFER-SOTA-013-002: CPU-Only llama.cpp Blocks V3 Gate

**Given** CUDA-capable torch and a mandated GGUF are available
**But** `llama_cpp.llama_cpp.llama_supports_gpu_offload()` is false
**When** Exp 2862 runs
**Then** `sota_runtime_ready_v3` is false
**And** the artifact documents the CUDA rebuild command instead of treating a
CPU-only load as headline runtime evidence.

### SCENARIO-INFER-SOTA-013-003: Missing Pair Is Recorded Separately

**Given** exactly one mandated SOTA GGUF exists locally
**When** `cached_sota_pair(gpu_indices=(0, 1))` returns `None`
**Then** `cached_sota_pair_returned_two_loadable_specs` is false
**And** the artifact still records the exact missing mandated model IDs and
does not use legacy small models to satisfy the pair or runtime gate.

### REQ-INFER-SOTA-3338: Exp 3338 SOTA GGUF Tokenizer Runtime Receipt

The system SHALL provide a standalone Exp 3338 preflight receipt before energy
descent reruns spend local SOTA inference. The receipt SHALL write
`results/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.json`; SHALL
define `MODEL_SPECS` through `cached_sota_pair(gpu_indices=(0, 1))` when that
resolver can produce a pair; SHALL still include all three mandated model IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; and SHALL NOT modify
`scripts/research_conductor.py`.

The receipt SHALL inspect local cache paths, file sizes, and checksum evidence
when files are present; SHALL record tokenizer dependency availability,
`llama_cpp` loader import metadata, and GPU visibility through the selected
Python's `torch.cuda` rather than relying only on `nvidia-smi`; and SHALL
attempt the smallest practical load/tokenize/generate smoke for each locally
available mandated GGUF. If a model cannot load, tokenize, or generate, the
receipt SHALL record the exact exception and continue to the next available
mandated model.

The artifact SHALL expose `honest_verdict`, `inference_substrate`,
`random_seed`, `reproducibility_checksum`, `duration_s`, `files_updated`,
`model_specs`, `cache_status`, `tokenizer_status`, `loader_status`,
`gpu_status`, `smoke_generation_status`, `runtime_receipt_clean`, and
`blocked_reasons`. `runtime_receipt_clean` SHALL be true only when at least one
mandated local SOTA GGUF completes a non-empty local load/tokenize/generate
smoke. Legacy small models MAY be used only as CPU loader controls, SHALL be
labeled non-headline, and SHALL never set `runtime_receipt_clean=true`.

### SCENARIO-INFER-SOTA-3338-001: Mandated Runtime Receipt Opens Gate

**Given** at least one mandated SOTA GGUF resolves from local cache
**And** the selected Python can import `llama_cpp` and sees CUDA through
`torch.cuda`
**When** Exp 3338 runs a bounded load/tokenize/generate smoke
**Then** `runtime_receipt_clean` is true only if a mandated model emits a
non-empty response
**And** the artifact records cache, tokenizer, loader, GPU, checksum, and
per-model smoke evidence for all mandated IDs.

### SCENARIO-INFER-SOTA-3338-002: Runtime Failures Block Precisely

**Given** mandated GGUF cache paths may exist locally
**But** every available mandated model fails tokenizer, loader, or generation
smoke
**When** Exp 3338 runs
**Then** `runtime_receipt_clean` is false
**And** `blocked_reasons` preserves the exact failure for each attempted
mandated model
**And** any legacy CPU smoke control remains non-headline.

### REQ-INFER-SOTA-3344: Exp 3344 Constrained Output Extractor llguidance Smoke

The system SHALL provide an Exp 3344 constrained structured-output extraction smoke for local SOTA GGUF outputs.
The artifact SHALL write `results/experiment_3344_constrained_output_extractor_llguidance_smoke_v1.json`.
The runner SHALL use `cached_sota_pair(gpu_indices=(0, 1))` to define `MODEL_SPECS`, requiring at least one mandated model.
It SHALL check for `llguidance` or `xgrammar` and implement a blocked artifact with exact missing dependency information if neither is available.
If available, it SHALL run a live extraction smoke on prompts requiring structured claims/constraints, comparing unconstrained to constrained output.
It SHALL measure `parse_failure_rate`, `schema_valid_rate`, `exact_verifier_accept_rate`, and `semantic_false_accept_count`.
Formatting success alone SHALL NOT be counted as correctness.
The artifact SHALL expose `honest_verdict`, `inference_substrate`, `random_seed`, `reproducibility_checksum`, `duration_s`, `files_updated`, `model_specs`, `constrained_tool`, `n_cases`, `parse_failure_rate_unconstrained`, `parse_failure_rate_constrained`, `exact_verifier_accept_rate_unconstrained`, `exact_verifier_accept_rate_constrained`, `semantic_false_accept_count`, `constrained_extractor_ready`, and `blocked_reasons`.

### SCENARIO-INFER-SOTA-3344-001: Missing Dependencies Block Precisely

**Given** neither llguidance nor xgrammar is installed
**When** the smoke test is run
**Then** it writes a blocked artifact rather than silently falling back to ad hoc parsing
**And** the output includes missing dependency reasons in `blocked_reasons`
**And** `constrained_extractor_ready` is false.

### REQ-INFER-SOTA-014: Exp 2870 SOTA Energy Baseline Micro-Panel

The system SHALL provide an Exp 2870 live SOTA GGUF micro-panel artifact that
runs only after
`results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json` reports
`sota_runtime_ready_v3=true`.  The runner SHALL call
`cached_sota_pair(gpu_indices=(0, 1))` first, SHALL use no legacy small model
for headline evidence, SHALL include at least one mandated SOTA GGUF from
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF` in the live `model_specs`, and SHALL select a
deterministic 10-20 row local micro-panel from `data/eval_manifests/` or FoVer
data.  It SHALL generate bounded greedy responses, capture first-token
logprob/top-k telemetry when the runtime exposes it, compute first-token
confidence and spilled/marginalized-energy-style logit baselines only from real
telemetry, and SHALL record `blocked_logprobs_unavailable` instead of
fabricating metrics when logits/logprobs are unavailable.

The artifact SHALL be written to
`results/experiment_2870_sota_energy_baseline_micro_panel_v1.json`; SHALL make
no full-benchmark claim; and SHALL expose `honest_verdict`,
`micro_panel_ready`, `live_model_invoked`, `model_specs`, `models_used`,
`n_examples`, `first_token_confidence_available`, `spilled_energy_available`,
`first_token_confidence_auroc`, `spilled_energy_auroc`,
`usable_response_count`, `blocked_metrics`, `sample_rows`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `field_principles`,
`run_date`, and `duration_s`.

### SCENARIO-INFER-SOTA-014-001: Live Micro-Panel Reports Cheap Logit Signals

**Given** Exp 2862 reports `sota_runtime_ready_v3=true`
**And** a mandated local SOTA GGUF model is resolved from `cached_sota_pair()`
or from the Exp 2862 selected model evidence
**When** Exp 2870 runs a deterministic 10-20 example micro-panel with bounded
greedy decoding
**Then** the artifact records live model invocation, model provenance, sample
rows, response correctness labels, first-token confidence availability, spilled
energy availability, AUROC values when both labels and telemetry are usable,
and explicit blocked metrics for unsupported telemetry or undefined AUROC.

### REQ-INFER-SOTA-015: Exp 2874 SOTA Runtime Clean Corrigendum V4

The system SHALL provide an Exp 2874 clean runtime corrigendum artifact that
records citation-grade local SOTA GGUF runtime state before downstream live
model tasks resume.  The artifact SHALL be written to
`results/experiment_2874_sota_runtime_clean_corrigendum_v4.json`; SHALL inspect
`.venv/bin/python` torch/CUDA availability, NVIDIA GPU inventory,
`llama_cpp` import/version/origin, `llama_cpp.llama_cpp.llama_supports_gpu_offload()`,
local cache resolution for `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, and
`cached_sota_pair(gpu_indices=(0, 1))`; SHALL run a bounded fixed-seed prompt
suite only when at least one mandated local GGUF, CUDA, and llama.cpp GPU
offload are all available; and SHALL NOT modify `scripts/research_conductor.py`
or `research-roadmap.yaml`.

The artifact SHALL expose `honest_verdict`, `sota_runtime_clean`,
`sota_runtime_ready_v4`, `model_specs`, `selected_model_hf_id`,
`selected_model_path`, `selected_model_checksum_or_fingerprint`,
`cached_sota_pair_returned_two_loadable_specs`,
`llama_cpp_gpu_offload_verified`, `preconditions_checked`, `prompt_suite`,
`usable_response_count`, `nonempty_response_count`, `total_tokens_generated`,
`tokens_per_second`, `gpu_memory_evidence`,
`legacy_small_models_used_only_for_smoke`, `random_seed`,
`reproducibility_checksum`, `tests_run`, `field_principles`, `run_date`, and
`duration_s`.  `sota_runtime_clean` and `sota_runtime_ready_v4` SHALL be true
only when at least one mandated SOTA GGUF generated a non-empty usable response
through GPU-backed llama.cpp with complete model fingerprint, prompt text,
timing, token, and GPU-memory provenance.  Legacy Qwen3.5-0.8B and Gemma4-E4B
models MAY appear only as CPU-smoke context and SHALL never set
`sota_runtime_clean=true`.

Pair readiness SHALL remain separate from single-model runtime readiness:
`cached_sota_pair_returned_two_loadable_specs` SHALL be true only when two
mandated local GGUF specs are actually loadable, but it SHALL NOT be required
for `sota_runtime_clean=true` when one mandated model has clean GPU-backed
runtime provenance.

### SCENARIO-INFER-SOTA-015-001: One Mandated Clean GPU Response Opens V4

**Given** CUDA-capable torch, NVIDIA GPUs, and llama.cpp GPU offload are
available
**And** exactly one mandated local SOTA GGUF resolves from cache
**When** Exp 2874 runs its fixed-seed prompt suite on that model
**Then** `sota_runtime_clean` and `sota_runtime_ready_v4` are true only if the
artifact records non-empty usable output text, generated-token count,
wall-clock timing, model fingerprint, GPU memory before/after evidence, and the
exact prompt text for at least one suite row.

### SCENARIO-INFER-SOTA-015-002: Missing Runtime Preconditions Block V4

**Given** any required single-model runtime precondition is unavailable
**When** Exp 2874 runs
**Then** it writes a blocked artifact with `sota_runtime_clean=false`,
`sota_runtime_ready_v4=false`, a populated `preconditions_checked` list, and no
legacy small-model promotion.

### SCENARIO-INFER-SOTA-015-003: Pair Readiness Is Not Single-Model Readiness

**Given** exactly one mandated local SOTA GGUF is cached and
`cached_sota_pair(gpu_indices=(0, 1))` returns `None`
**When** Exp 2874 records a clean single-model GPU runtime
**Then** `cached_sota_pair_returned_two_loadable_specs` remains false while
`sota_runtime_clean` may be true.

### REQ-INFER-SOTA-016: Exp 2875 SOTA Energy/Logprob Micro-Panel Corrigendum V2

The system SHALL provide an Exp 2875 corrigendum artifact for the Exp 2870
SOTA energy micro-panel telemetry gap.  The artifact SHALL be written to
`results/experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.json`;
SHALL first confirm that Exp 2874 reports `sota_runtime_clean=true`, the
selected model is one of the mandated headline GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`, the selected GGUF path exists, and
llama.cpp GPU offload is visible; and SHALL then verify that the live llama.cpp
path exposes token logprobs or a documented substitute telemetry path before
running the tiny prompt panel.

The runner SHALL use only mandated headline GGUFs in headline prompt rows.
Legacy Qwen3.5-0.8B and Gemma4-E4B models MAY appear only as CPU-smoke context
and SHALL never set `micro_panel_clean=true`.  The prompt panel SHALL use fixed
seeded, bounded prompts with existing FEVER/FoVer/HaluEval-style labels only;
it SHALL record raw outputs, non-empty response status, first-token confidence
when token logprobs are present, spilled-energy-style score from token
logprobs, or a documented substitute score when substitute telemetry is used.

The artifact SHALL expose `honest_verdict`, `micro_panel_clean`,
`blocked_reason`, `model_specs`, `selected_model_hf_id`,
`selected_model_path`, `preconditions_checked`, `n_prompts`,
`n_nonempty_responses`, `logprobs_available`, `substitute_telemetry_used`,
`prompt_rows`, `auroc_if_computable`, `benchmark_claim_made`, `random_seed`,
`tests_run`, `field_principles`, `run_date`, and `duration_s`.
`micro_panel_clean` SHALL be true only when every prompt row has a non-empty
response and every prompt row has sufficient real token-logprob or documented
substitute telemetry for the stated diagnostic analysis.  If telemetry is
unavailable, the artifact SHALL set `blocked_reason` to
`blocked_logprobs_unavailable` or the exact `blocked_*` precondition verdict and
SHALL make no benchmark claim.

### SCENARIO-INFER-SOTA-016-001: Clean Telemetry Allows Diagnostic Panel

**Given** Exp 2874 is clean, the selected mandated GGUF path exists, GPU offload
is visible, and llama.cpp exposes token logprobs or documented substitute
telemetry
**When** Exp 2875 runs the fixed tiny micro-panel
**Then** the artifact records all prompt rows with non-empty responses, model
provenance, telemetry-derived scores, AUROC only when label variance makes it
computable, `micro_panel_clean=true`, and `benchmark_claim_made=false`.

### SCENARIO-INFER-SOTA-016-002: Missing Telemetry Blocks Corrigendum

**Given** Exp 2874 is clean and the selected mandated GGUF can generate text
**But** llama.cpp exposes neither token logprobs nor documented substitute
telemetry
**When** Exp 2875 runs
**Then** it writes a complete blocked artifact with
`blocked_reason=blocked_logprobs_unavailable`, preserves the exact precondition
evidence, keeps `prompt_rows` diagnostic-only, sets `micro_panel_clean=false`,
and makes no benchmark claim.

### REQ-INFER-SOTA-017: Exp 2886 SOTA Micro-Panel Clean Telemetry V3

The system SHALL provide an Exp 2886 corrigendum that clears the .272 capstone's
adversarial flag on the Exp 2875 micro-panel.  The artifact SHALL be written to
`results/experiment_2886_sota_micro_panel_clean_telemetry_v3.json`; SHALL reuse
the REQ-INFER-SOTA-016 preconditions; and SHALL additionally expose a
`reproducibility_checksum`, an `selected_model_fingerprint`, a
`cached_sota_pair_returned_two_loadable_specs` flag read from Exp 2874, a
`gpu_memory_evidence` object with nvidia-smi snapshots taken before and after the
panel runs, a `micro_panel_downgraded_to_non_benchmark` boolean, and an
`adversarial_verify_invoked` boolean.  The artifact SHALL re-load itself through
`scripts.adversarial_verify.verify_artifact` and embed the result under
`adversarial_verify_result`.  The artifact's wall-clock `duration_s` SHALL be the
genuine measured duration of the panel run, not a sleep-padded value.
`micro_panel_clean` SHALL be true only when every row has non-empty text and real
token-logprob or documented substitute telemetry; otherwise the artifact SHALL
set `micro_panel_downgraded_to_non_benchmark=true` and `benchmark_claim_made=false`.

### SCENARIO-INFER-SOTA-017-001: Clean Telemetry Survives Adversarial Verify

**Given** Exp 2874 reports `sota_runtime_clean=true`, llama.cpp exposes token
logprobs, and the panel runner returns non-empty rows with finite logprobs
**When** Exp 2886 v3 builds the artifact
**Then** the artifact records `micro_panel_clean=true`,
`micro_panel_downgraded_to_non_benchmark=false`, a non-null
`reproducibility_checksum`, a populated `gpu_memory_evidence`, and an embedded
`adversarial_verify_result`.

### SCENARIO-INFER-SOTA-017-002: Missing Telemetry Triggers Downgrade

**Given** the panel rows lack token logprobs and lack documented substitute
telemetry
**When** Exp 2886 v3 runs
**Then** the artifact records `micro_panel_clean=false`,
`micro_panel_downgraded_to_non_benchmark=true`, an honest_verdict starting with
`complete:`, and makes no benchmark claim.

### SCENARIO-INFER-SOTA-017-003: Precondition Failure Blocks Before Panel

**Given** Exp 2874 reports `sota_runtime_clean=false`
**When** Exp 2886 v3 runs
**Then** the artifact records a precondition-derived `blocked_reason`, leaves
the panel rows empty, populates the reproducibility checksum from the empty
panel inputs, and still embeds the adversarial verify result.

### REQ-INFER-SOTA-018: Exp 2917 Spilled-Energy Logit Detector Micro-Panel

The system SHALL provide an Exp 2917 diagnostic micro-panel runner that writes
`results/experiment_2917_spilled_energy_logit_detector_micro_panel_v1.json`.
The runner SHALL call `cached_sota_pair(gpu_indices=(0, 1))` before any fallback
model resolution, SHALL use only locally cached mandated SOTA GGUF model specs
for live llama.cpp inference, and SHALL include at least one of
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF` in the live path.  If the runtime returns no
token logprobs, top-k logprobs, or final-token logits, the runner SHALL write a
complete artifact with `blocked_reason=blocked_logprob_runtime_unavailable`,
`spilled_energy_micro_panel_ready=false`, `logprob_or_logits_available=false`,
and no benchmark claim.

When telemetry is available, the runner SHALL build a bounded 24-40 example
panel from local verified/failing code-generation outputs and local factuality
examples.  Every example row SHALL include source provenance, prompt hash, raw
LLM response, model id, token count, logprob/logit availability, verification
label, final-token spilled-energy score, final-token marginal-energy score, and
sequence-level spilled/marginal energy summaries when token logprobs are
present.  The runner SHALL compute only simple diagnostic separation summaries
(for example group means, deltas, and tie-aware AUROC over fixed features) and
SHALL NOT train a detector or promote any benchmark/matrix-row claim.

The artifact SHALL expose `honest_verdict`,
`spilled_energy_micro_panel_ready`, `benchmark_claim_made`, `claim_boundary`,
`model_specs`, `models_used`, `cached_sota_pair_used`, `examples`,
`random_seed`, `logprob_or_logits_available`, `spilled_energy_features`,
`separation_summary`, `inference_substrate`, `duration_s`, and `run_date`.
`claim_boundary` SHALL equal `diagnostic_only_no_benchmark_claim`;
`benchmark_claim_made` SHALL always be false; `random_seed` SHALL equal 2917;
`inference_substrate` SHALL equal `live_llm_inference`; and `run_date` SHALL
equal `20260523`.

### SCENARIO-INFER-SOTA-018-001: Logprob Runtime Writes Diagnostic Micro-Panel

**Given** at least one mandated SOTA GGUF can be resolved after
`cached_sota_pair(gpu_indices=(0, 1))` is attempted
**And** the live inference runtime returns token logprobs, top-k logprobs, or
final-token logits for panel prompts
**When** Exp 2917 runs
**Then** it writes a 24-40 row diagnostic artifact with raw responses,
prompt hashes, model provenance, verification labels, spilled/marginal energy
features, simple separation metrics, `claim_boundary` set to
`diagnostic_only_no_benchmark_claim`, and `benchmark_claim_made=false`.

### SCENARIO-INFER-SOTA-018-002: Missing Logprob Runtime Blocks Cleanly

**Given** at least one mandated SOTA GGUF can generate text
**But** the runtime returns neither token logprobs, top-k logprobs, nor
final-token logits
**When** Exp 2917 runs
**Then** it writes `blocked_reason=blocked_logprob_runtime_unavailable`,
`spilled_energy_micro_panel_ready=false`, `logprob_or_logits_available=false`,
and exits without fabricating spilled-energy metrics or benchmark claims.

### REQ-INFER-SOTA-019: Exp 2989 SOTA GGUF Cache Provenance Preflight

The system SHALL provide an Exp 2989 bounded SOTA GGUF cache/provenance
preflight that gates downstream headline live-LLM claims before Exp 2977-style
repair evidence is reused.  The preflight SHALL be written to
`results/experiment_2989_sota_gguf_cache_provenance_preflight_v1.json`; SHALL
record CUDA/GPU availability, free VRAM when available, the repository commit,
the Python environment, and local cache-root existence before any model load;
SHALL inspect the mandated headline GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through the shared `cached_sota_pair()` /
local-cache resolver pattern; and SHALL attempt a bounded conservative
llama.cpp prompt only for locally available headline GGUF files.

For every headline model inspected, the artifact SHALL record the exact model
ID, local cache path when present, file-size/checksum evidence when feasible,
load status, generation status, duration, and transcript path/hash when live
generation succeeds.  Legacy small models `Qwen/Qwen3.5-0.8B` and
`unsloth/gemma-4-E4B-it-GGUF` MAY appear only as smoke-only context and SHALL
never set `sota_headline_ready=true`.

The artifact SHALL expose `sota_headline_ready`, `preconditions_checked`,
`model_specs`, `sota_models_attempted`, `sota_models_available`, `cache_paths`,
`model_checksums`, `live_transcript_paths`, `legacy_smoke_only_used`,
`inference_substrate`, `duration_seconds`, and `honest_verdict`.
`sota_headline_ready` SHALL be true only when at least one mandated headline
GGUF produces a real non-empty transcript with cache/provenance fields.

### SCENARIO-INFER-SOTA-019-001: Available Headline GGUF Produces Transcript Evidence

**Given** preconditions have been recorded before model loading
**And** at least one mandated headline GGUF resolves to a local file
**When** Exp 2989 runs a bounded conservative prompt on that headline model
**Then** `sota_headline_ready=true` only if the model produces a non-empty
transcript
**And** the artifact records the transcript path/hash, model ID, cache path,
checksum evidence, load status, generation status, measured duration, and
`honest_verdict` starting with `success:`.

### SCENARIO-INFER-SOTA-019-002: Missing Headline Cache Blocks Without Small-Model Promotion

**Given** no mandated headline GGUF resolves to a local file
**When** Exp 2989 runs
**Then** it writes a terminal blocked artifact with `sota_headline_ready=false`
**And** all mandated headline IDs appear in `sota_models_attempted` with
missing-cache status
**And** any legacy small-model smoke context is labeled `smoke_only` and never
contributes to headline readiness.

### REQ-INFER-SOTA-020: Exp 3001 SOTA GGUF Cache Carry-Forward Checksum Refresh

The system SHALL provide an Exp 3001 local-only cache/provenance refresh that
gates milestone `.282` headline live-LLM claims on fresh SOTA GGUF evidence
rather than reused Exp 2989 state.  The refresh SHALL be written to
`results/experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json`;
SHALL record CUDA/GPU availability, free VRAM when available, repository
commit, Python environment, cache roots, and checksum feasibility before any
model load; SHALL inspect the mandated headline GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through the shared `cached_sota_pair()` /
local-cache resolver pattern; and SHALL attempt bounded conservative
llama.cpp generation only for locally available headline GGUF files.

For every headline model inspected, the artifact SHALL record exact model ID,
local cache path when present, file-size/checksum evidence when feasible, load
status, generation status, measured duration, and transcript path/hash when
live generation succeeds.  Legacy small models `Qwen/Qwen3.5-0.8B` and
`unsloth/gemma-4-E4B-it-GGUF` MAY appear only as smoke-only context and SHALL
never set `sota_headline_ready=true`.

The artifact SHALL expose `sota_headline_ready`, `preconditions_checked`,
`model_specs`, `sota_models_attempted`, `sota_models_available`, `cache_paths`,
`model_checksums`, `live_transcript_paths`, `legacy_smoke_only_used`,
`inference_substrate`, `duration_seconds`, and `honest_verdict`.
`sota_headline_ready` SHALL be true only when at least one mandated headline
GGUF produces a real non-empty transcript with cache/provenance fields.

### SCENARIO-INFER-SOTA-020-001: Fresh Headline GGUF Transcript Opens The .282 Gate

**Given** Exp 3001 has recorded preconditions before model loading
**And** at least one mandated headline GGUF resolves to a local file
**When** Exp 3001 runs a bounded conservative prompt on that headline model
**Then** `sota_headline_ready=true` only if the model produces a non-empty
transcript
**And** the artifact records transcript path/hash, model ID, cache path,
checksum evidence, load status, generation status, measured duration, and an
`honest_verdict` starting with `success:`.

### SCENARIO-INFER-SOTA-020-002: No Headline Run Emits Terminal Blocked Artifact

**Given** no mandated headline GGUF can complete live generation locally
**When** Exp 3001 runs
**Then** it writes a terminal blocked artifact with `sota_headline_ready=false`
**And** all mandated headline IDs appear in `sota_models_attempted` with
auditable missing-cache, precondition-failure, or generation-failure status
**And** any legacy small-model smoke context is labeled smoke-only and never
contributes to headline readiness.

### REQ-INFER-SOTA-021: Exp 3013 SOTA GGUF Logprob Telemetry Preflight

The system SHALL provide an Exp 3013 local-only SOTA GGUF readiness refresh
that determines both whether at least one mandated headline GGUF can produce a
live transcript and whether the current local loader exposes token logprob and
top-k telemetry suitable for Cactus-style constrained acceptance.  The
preflight SHALL be written to
`results/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json`;
SHALL record CUDA/GPU availability, free VRAM when available, repository
commit, Python environment, cache roots, and checksum feasibility before any
model load; SHALL inspect the mandated headline GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through the shared `cached_sota_pair()` /
local-cache resolver pattern; and SHALL attempt bounded conservative
llama.cpp generation only for locally available headline GGUF files.

For every headline model inspected, the artifact SHALL record exact model ID,
local cache path when present, file-size/checksum evidence when feasible, load
status, generation status, measured duration, transcript path/hash when live
generation succeeds, and observed loader telemetry capability booleans for
token logprobs, top-k alternatives, and logits.  Legacy small models
`Qwen/Qwen3.5-0.8B` and `unsloth/gemma-4-E4B-it-GGUF` MAY appear only as
smoke-only context and SHALL never set `sota_headline_ready=true`.

The artifact SHALL expose `sota_headline_ready`, `sota_logprob_ready`,
`preconditions_checked`, `model_specs`, `headline_models_attempted`,
`headline_models_available`, `telemetry_capabilities`, `cache_paths`,
`model_checksums`, `live_transcript_paths`, `legacy_smoke_only_used`,
`inference_substrate`, `duration_s`, and `honest_verdict`.
`sota_headline_ready` SHALL be true only when at least one mandated headline
GGUF produces a real non-empty transcript with cache/provenance fields.
`sota_logprob_ready` SHALL be true only when observed live loader output
contains token logprob and top-k alternative data; live transcript generation
without telemetry SHALL leave `sota_logprob_ready=false` without fabricating
telemetry.

The Exp 3013 runner SHALL also support the v466 runtime-truth artifact shape
used by downstream uPRM/VPR/DCCD/D6 gates.  The artifact SHALL resolve all
three mandated GGUFs through the local cache or shared `cached_sota_pair()` /
resolver pattern, SHALL NOT call `AutoTokenizer` on a `*-GGUF` repo ID, SHALL
probe configured llama.cpp-compatible endpoints such as
`http://127.0.0.1:8080` for completion plus top-logprob or confidence
telemetry, and SHALL expose `usable_sota_models`, `sota_models_ready`,
`completion_endpoint_ready`, `logprob_endpoint_ready`,
`top_logprob_or_confidence_ready`, `tool_first_verifier_ready`,
`live_completion_invoked`, `skip_reasons`, and `flagged_adversarial`.
When no endpoint returns a non-empty completion, the artifact SHALL set
`live_completion_invoked=false`, SHALL avoid the `live_llm_inference`
substrate declaration, and SHALL use a terminal-prefix verdict such as
`complete_gguf_logprob_preflight_partial_ready` or
`blocked_gguf_logprob_preflight_no_ready_paths`.  When a live endpoint
completion is invoked, the artifact SHALL record endpoint timing and enough
probe detail to distinguish a real endpoint call from a cache-only preflight.
For the v467 endpoint bring-up repair, the artifact SHALL also record
`server_command`, `endpoint_url`, `sample_completion`,
`sample_logprob_evidence`, and `blocker_root_cause`; SHALL attempt to start a
local llama.cpp-compatible `llama-server` on a free port when no configured
endpoint is already usable; SHALL prefer the smallest locally available
mandated GGUF for that bring-up while keeping all three mandated IDs in
`model_specs`; and SHALL emit a terminal-prefix verdict such as
`success_llamacpp_logprob_endpoint_ready` or
`blocked_llamacpp_logprob_endpoint_bringup_no_binary_or_runtime` without
claiming live inference when no completion was actually invoked.

### SCENARIO-INFER-SOTA-021-001: Live Transcript With Telemetry Opens Both Gates

**Given** Exp 3013 has recorded preconditions before model loading
**And** at least one mandated headline GGUF resolves to a local file
**When** Exp 3013 runs a bounded conservative prompt and the loader returns
non-empty text with token logprob and top-k alternative fields
**Then** `sota_headline_ready=true`
**And** `sota_logprob_ready=true`
**And** the artifact records transcript path/hash, model ID, cache path,
checksum evidence, load status, generation status, measured duration, and
observed telemetry capability fields.

### SCENARIO-INFER-SOTA-021-002: Live Transcript Without Telemetry Does Not Fake Logprobs

**Given** a mandated headline GGUF produces non-empty text locally
**But** the loader output lacks token logprobs or top-k alternatives
**When** Exp 3013 completes
**Then** `sota_headline_ready=true`
**And** `sota_logprob_ready=false`
**And** `telemetry_capabilities` records the missing telemetry blockers without
fabricating logprob, top-k, or logits evidence.

### SCENARIO-INFER-SOTA-021-003: No Headline Run Emits Blocked SOTA Artifact

**Given** no mandated headline GGUF can complete live generation locally
**When** Exp 3013 runs
**Then** it writes a terminal blocked artifact with
`honest_verdict=blocked_sota_headline_model_unavailable`
**And** all mandated headline IDs appear in `headline_models_attempted` with
auditable missing-cache, precondition-failure, or generation-failure status
**And** any legacy small-model smoke context is labeled smoke-only and never
contributes to headline readiness or logprob readiness.

### SCENARIO-INFER-SOTA-021-004: Endpoint Runtime Truth Is Split From Cache Readiness

**Given** all three mandated GGUFs resolve to local `.gguf` files
**And** no configured llama.cpp endpoint returns a completion
**When** Exp 3013 writes the v466 runtime-truth artifact
**Then** `sota_models_ready=true`
**And** `completion_endpoint_ready=false`
**And** `live_completion_invoked=false`
**And** `inference_substrate` does not claim `live_llm_inference`
**And** `skip_reasons` records the missing endpoint and telemetry lanes.

**Given** a configured llama.cpp endpoint returns a non-empty completion with
top-logprob telemetry
**When** Exp 3013 writes the v466 runtime-truth artifact
**Then** `completion_endpoint_ready=true`
**And** `logprob_endpoint_ready=true`
**And** `top_logprob_or_confidence_ready=true`
**And** the artifact records endpoint timing and a terminal
`complete_gguf_logprob_preflight_ready` verdict.

### SCENARIO-INFER-SOTA-021-005: Llama.cpp Endpoint Bring-Up Emits Runtime Evidence

**Given** all three mandated GGUFs resolve to local `.gguf` files
**And** no configured llama.cpp-compatible endpoint returns a completion
**When** Exp 3013 writes the v467 endpoint bring-up artifact
**Then** it records CUDA/GPU visibility, llama.cpp Python and server-binary
availability, local GGUF paths, a free-port probe, and relevant environment
variables in `preconditions_checked`
**And** it either starts `llama-server` with the smallest operational mandated
GGUF and records `server_command`, PID, endpoint URL, server logs, cleanup
behavior, sample completion, and token/top-logprob evidence
**Or** it records `live_completion_invoked=false`, null sample telemetry, and
`blocker_root_cause` with exact missing binary, runtime, port, or server-log
evidence
**And** the terminal verdict starts with
`success_llamacpp_logprob_endpoint_ready` only when completion plus logprob or
top-logprob telemetry were observed, otherwise with
`blocked_llamacpp_logprob_endpoint_bringup_`.

### REQ-INFER-SOTA-022: Exp 3043 Transcript Fingerprint Replay Preflight

The system SHALL provide an Exp 3043 local-only transcript fingerprint
preflight for deterministic repair-inference replay.  The preflight SHALL write
`results/experiment_3043_verified_speculation_transcript_fingerprint_v1.json`;
SHALL resolve mandated local SOTA GGUFs through `cached_sota_pair()` or the
shared SOTA local-cache resolver pattern; SHALL NOT use legacy small models as
headline evidence; and SHALL attempt live llama.cpp generation on at least one
available mandated GGUF from `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`.

Before model loading, the preflight SHALL record CUDA availability, GPU free
memory when available, repository commit, Python executable/version, cache
resolution for every mandated GGUF, selected model path, model checksum
feasibility, seed, batch size, decoding parameters, and wall-clock duration.
If no mandated GGUF is locally loadable, the artifact SHALL set
`fingerprint_live_ready=false`, preserve the precondition evidence, and emit a
`blocked_sota_gguf_unavailable` verdict without falling back to legacy models.

For a tiny fixed repair-style prompt set, the preflight SHALL record
`prompt_hash`, `model_hash_or_cache_path`, `decode_config`, `seed`,
`raw_output_hash`, `normalized_output_hash`, and `batch_context_hash` for each
transcript.  It SHALL rerun the same prompt/config at least once and set
`deterministic_replay_passed` only when the raw and normalized output hashes
match across runs.  Runtime nondeterminism SHALL be recorded as explicit replay
divergence rather than hidden.

The artifact SHALL expose `fingerprint_live_ready`, `models_used`,
`model_specs`, `legacy_smoke_only_used`, `n_prompts`,
`deterministic_replay_passed`, `prompt_hashes`, `output_hashes`,
`decode_config`, `batch_context_hash`, `reproducibility_checksum`,
`inference_substrate`, and `honest_verdict`.  It SHALL make no repair
pass-rate, performance, or promotion claim.

### SCENARIO-INFER-SOTA-022-001: Deterministic Transcript Replay Opens Fingerprint Gate

**Given** at least one mandated SOTA GGUF resolves locally
**And** the local runtime returns identical text hashes for repeated bounded
repair-style prompts under the same seed and decode config
**When** Exp 3043 runs
**Then** `fingerprint_live_ready=true`
**And** `deterministic_replay_passed=true`
**And** the artifact records prompt hashes, output hashes, model/cache
identity, decode config, batch context hash, reproducibility checksum, and
pre-load inference substrate metadata.

### SCENARIO-INFER-SOTA-022-002: Replay Divergence Is Preserved

**Given** a mandated SOTA GGUF resolves locally and generates text
**But** repeated runs under the same prompt, seed, and decode config produce
different raw or normalized output hashes
**When** Exp 3043 completes
**Then** `deterministic_replay_passed=false`
**And** `fingerprint_live_ready=false`

### REQ-INFER-SOTA-023: Robust Reusable GGUF Generator Harness

The system SHALL expose `python/carnot/verify/gguf_inference.py` as the shared
live GGUF generation entrypoint for verifier self-check experiments.  The
entrypoint SHALL provide `load_gguf_generator(prefer_order=None, n_ctx=1024,
max_n_gpu_layers=-1)` and `generate(generator, prompt, max_tokens)`.

`load_gguf_generator` SHALL resolve only locally cached headline GGUF files for
`unsloth/gemma-4-26B-A4B-it-GGUF`,
`unsloth/Qwen3.6-35B-A3B-GGUF`, and
`unsloth/gemma-4-31B-it-GGUF`, in that default order unless an explicit
`prefer_order` is supplied.  For each candidate, it SHALL try llama.cpp loading
with `n_gpu_layers` values `[max_n_gpu_layers, 20, 0]`, catching and recording
every load or generation exception before moving to the next fallback.

After each successful load, the harness SHALL run a bounded one-token
generation smoke prompt and require at least one emitted output token before it
returns readiness.  The returned metadata SHALL include `model_used`,
`gguf_path`, `n_gpu_layers_used`, `load_s`, `smoke_s`, `smoke_tokens`, and
`fallback_index`.  If every candidate/offload combination fails, the harness
SHALL raise a clear `RuntimeError` that names the failed model/offload attempts
so callers can emit an honest `blocked_all_gguf_inference_failed` verdict.

### SCENARIO-INFER-SOTA-023-001: Cached Headline GGUF Loads And Smokes

**Given** CUDA, llama.cpp, and at least one mandated headline GGUF are available
locally
**When** `load_gguf_generator()` is called
**Then** it returns a llama.cpp-backed generator plus metadata naming the
selected model, GGUF path, offload level, fallback index, and positive smoke
token count
**And** a second `generate(...)` call returns non-empty text.

### SCENARIO-INFER-SOTA-023-002: Exhausted Fallbacks Report Honest Failure

**Given** every candidate model/offload attempt fails to load or emit a smoke
token
**When** `load_gguf_generator()` exhausts its fallback chain
**Then** it raises `RuntimeError` with one failure entry per attempted
model/offload combination instead of returning a fabricated generator.
**And** the artifact records replay divergence rows instead of claiming
deterministic readiness.

### SCENARIO-INFER-SOTA-022-003: Missing SOTA GGUF Blocks Without Legacy Evidence

**Given** no mandated SOTA GGUF resolves from local cache
**When** Exp 3043 runs
**Then** it writes a terminal artifact with `fingerprint_live_ready=false`
**And** `legacy_smoke_only_used=false`
**And** `honest_verdict` starts with `blocked_sota_gguf_unavailable`.

### REQ-INFER-SOTA-023: Exp 3123 SOTA Cache Preconditions Manifest v2

The system SHALL provide an Exp 3123 local-only SOTA cache/precondition
manifest that downstream live-LLM headline tasks can consume before invoking
verifiers, repair loops, self-learning loops, energy sidecars, or any other
compute-bound live model path.  The manifest SHALL probe only local filesystem,
cache, GPU-preflight, and checked-in artifact state; it SHALL NOT download
models and SHALL NOT run live model inference.

The manifest SHALL write
`results/experiment_3123_sota_cache_preconditions_manifest_v2.json` and SHALL
make the mandated headline policy explicit for:
`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`.  It SHALL record nonzero local GGUF
availability separately from HuggingFace `.no_exist` sentinels and zero-byte
markers, and it SHALL preserve whether `cached_sota_pair()` can resolve two
loadable mandated specs separately from whether at least one mandated SOTA
model is available for a bounded single-model live attempt.

The artifact SHALL include `sota_cache_manifest_v2_ready`,
`mandatory_headline_model_ids`, `present_model_ids`, `missing_model_ids`,
`selected_headline_model_ids`, `cached_sota_pair_available`,
`any_single_sota_available`, `headline_claim_allowed`, `gpu_preflight`,
`downstream_usage`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`.  `headline_claim_allowed` SHALL be true only for downstream
work that attempts at least one locally present mandated model and records live
model evidence; pair/comparative headline claims SHALL additionally require
`cached_sota_pair_available=true`.

### SCENARIO-INFER-SOTA-023-001: Single Cached Mandated Model Opens Bounded Attempts Only

**Given** exactly one mandated SOTA GGUF resolves from local cache
**And** `cached_sota_pair()` does not return two loadable specs
**When** Exp 3123 builds the precondition manifest
**Then** `any_single_sota_available=true`
**And** `cached_sota_pair_available=false`
**And** `selected_headline_model_ids` contains only the present mandated model
**And** downstream live-LLM rules require a bounded attempt on that present
model or a blocked/diagnostic artifact before any headline claim.

### SCENARIO-INFER-SOTA-023-002: Missing Mandated Cache Blocks Live Claims Without Solver Work

**Given** no mandated SOTA GGUF resolves from local cache
**When** Exp 3123 builds the precondition manifest
**Then** `present_model_ids=[]`
**And** `missing_model_ids` lists all mandated headline model IDs
**And** `headline_claim_allowed=false`
**And** solver-only downstream rules remain allowed without model availability
while live-LLM downstream rules require a blocked/diagnostic artifact.

### SCENARIO-INFER-SOTA-023-003: Legacy Small Models Are Smoke-Only

**Given** legacy small model cache entries exist
**When** Exp 3123 builds the precondition manifest
**Then** legacy small model IDs may appear only in `smoke_test_model_ids`
**And** they SHALL NOT appear in `present_model_ids`,
`selected_headline_model_ids`, or any headline-claim allowance rule.

### REQ-INFER-SOTA-024: Exp 3206 CUDA Environment Forensics Ledger

The system SHALL provide an Exp 3206 CUDA environment ledger before any
downstream llama.cpp rebuild or full local SOTA receipt rerun.  The ledger
SHALL run clean subprocess probes with the selected Python interpreter, SHALL
import `torch` before any project imports in one clean subprocess, SHALL import
`llama_cpp` in a separate clean subprocess, and SHALL avoid loading a large
GGUF model while probing llama.cpp GPU-offload support.

The artifact SHALL be written to
`results/experiment_3206_cuda_env_forensics_ledger_v1.json` and SHALL record
`nvidia-smi`, `nvcc --version` when present, NVIDIA driver/device/memory
state, selected Python executable, virtualenv, selected interpreter `sys.path`,
`pip show torch`, `pip show llama-cpp-python`, `llama_cpp` filesystem origin,
`CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`, `PATH`, `CMAKE_ARGS`,
`FORCE_CMAKE`, and llama.cpp/ggml/CUDA-related environment variables.

The artifact SHALL expose at least `schema_version`, `experiment_id`,
`milestone`, `selected_python`, `virtualenv`, `nvidia_smi_available`,
`gpu_count_nvidia_smi`, `torch_version`, `torch_cuda_version`,
`torch_cuda_available_clean_subprocess`,
`torch_cuda_device_count_clean_subprocess`, `llama_cpp_version`,
`llama_cpp_origin`, `llama_cpp_cuda_build_detected`,
`clean_subprocess_stderr_tail`, `cuda_env_vars`, `cuda_env_diagnosed`,
`cuda_init_clean`, `recommended_next_action`, `conductor_file_modified`,
`active_roadmap_modified`, and `honest_verdict`.  `cuda_init_clean` SHALL be
true only when the selected Python can initialize torch CUDA, sees at least one
CUDA device, llama_cpp imports, llama.cpp reports GPU offload support, and the
clean subprocess stderr does not report a CUDA initialization failure.

### SCENARIO-INFER-SOTA-024-001: Clean CUDA Stack Allows Receipt Rerun

**Given** `nvidia-smi` reports at least one NVIDIA GPU
**And** the selected Python clean torch subprocess reports CUDA available and a
nonzero device count
**And** the clean llama_cpp subprocess imports and reports GPU offload support
without CUDA initialization errors
**When** Exp 3206 builds the CUDA environment ledger
**Then** `cuda_env_diagnosed=true`
**And** `cuda_init_clean=true`
**And** `recommended_next_action` allows a full local SOTA receipt rerun.

### SCENARIO-INFER-SOTA-024-002: Torch CUDA Failure Blocks Full Receipt

**Given** `nvidia-smi` reports at least one NVIDIA GPU
**But** the selected Python clean torch subprocess reports
`torch.cuda.is_available() == false`
**When** Exp 3206 builds the CUDA environment ledger
**Then** `cuda_env_diagnosed=true`
**And** `cuda_init_clean=false`
**And** `recommended_next_action` keeps the next step on selected-Python CUDA
repair rather than a full receipt rerun.

### SCENARIO-INFER-SOTA-024-003: llama.cpp CUDA Init Failure Is Preserved

**Given** the clean llama_cpp subprocess emits
`ggml_cuda_init: failed to initialize CUDA`
**When** Exp 3206 builds the CUDA environment ledger
**Then** `llama_cpp_cuda_build_detected=true`
**And** `clean_subprocess_stderr_tail` preserves the stderr line
**And** `cuda_init_clean=false`.

### REQ-INFER-SOTA-025: Exp 3207 llama.cpp CUDA Rebuild Clean Subprocess Gate

The system SHALL provide an Exp 3207 runtime gate that reads the Exp 3206 CUDA
environment ledger before attempting any llama.cpp rebuild.  The gate SHALL
write `results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json`,
SHALL preserve Exp 3206's `recommended_next_action`, and SHALL NOT modify
`scripts/research_conductor.py` or `research-roadmap.yaml`.

If the selected Python clean subprocess still cannot initialize torch CUDA or
reports zero CUDA devices, Exp 3207 SHALL NOT rebuild llama.cpp blindly.  It
SHALL record `rebuild_attempted=false`, `torch_cuda_available_after=false`, a
human-readable blocker containing the torch/CUDA stderr, and SHALL keep
`cpu_fallback_only=true`, `cuda_receipt_ready=false`, and
`clean_rerun_allowed_candidate=false`.

If torch CUDA is available and the ledger indicates that llama.cpp CUDA support
or initialization is the remaining blocker, Exp 3207 MAY attempt the minimum
safe project-local rebuild or alternate binary path.  The rebuild command SHALL
use the selected interpreter's dependency-management path and the llama.cpp CUDA
configuration (`CMAKE_ARGS=-DGGML_CUDA=ON` with `FORCE_CMAKE=1`, or an
equivalent CUDA-enabled llama-cpp-python install path).  The artifact SHALL
record a bounded command summary and rebuild log tail.

After any justified rebuild or alternate binary selection, Exp 3207 SHALL run a
clean subprocess smoke probe that imports `llama_cpp`, detects CUDA-capable
llama.cpp build evidence, and verifies that GPU offload support can be queried
without an immediate `ggml_cuda_init` failure.  CPU fallback SHALL never count
as success.

The artifact SHALL expose `schema_version`, `experiment_id`, `milestone`,
`env_ledger_artifact`, `rebuild_attempted`, `rebuild_command_summary`,
`rebuild_log_tail`, `torch_cuda_available_after`,
`llama_cpp_cuda_build_detected_after`,
`clean_subprocess_gpu_offload_probe_passed`, `cpu_fallback_only`,
`cuda_receipt_ready`, `clean_rerun_allowed_candidate`, `blocker`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.

### SCENARIO-INFER-SOTA-025-001: Torch CUDA Blocker Stops Blind Rebuild

**Given** the Exp 3206 ledger recommends selected-Python torch CUDA repair
**And** the current clean torch subprocess still reports
`torch.cuda.is_available() == false`
**When** Exp 3207 runs
**Then** it does not invoke a llama.cpp rebuild command
**And** the artifact records the exact torch/CUDA blocker
**And** `cuda_receipt_ready=false`.

### SCENARIO-INFER-SOTA-025-002: Rebuild Is Justified Only After Torch CUDA Works

**Given** the current clean torch subprocess reports CUDA available with at
least one device
**And** llama.cpp imports but lacks GPU offload support or emits a CUDA init
blocker
**When** Exp 3207 runs
**Then** it records a CUDA-enabled llama-cpp-python rebuild or alternate binary
command summary
**And** preserves the rebuild log tail.

### SCENARIO-INFER-SOTA-025-003: Clean GPU Offload Probe Opens Receipt Candidate

**Given** torch CUDA is available after the rebuild decision
**And** the clean llama.cpp subprocess detects CUDA build evidence and no
`ggml_cuda_init` failure
**When** Exp 3207 completes
**Then** `clean_subprocess_gpu_offload_probe_passed=true`
**And** `cpu_fallback_only=false`
**And** `cuda_receipt_ready=true`
**And** `clean_rerun_allowed_candidate=true`.

### REQ-INFER-SOTA-026: Exp 3220 Hermetic CUDA Runtime Repair Ledger

The system SHALL provide an Exp 3220 hermetic CUDA runtime repair ledger that
writes `results/experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.json`
without loading a SOTA GGUF model or claiming model receipt.  The ledger SHALL
probe the selected repository Python in clean subprocesses with
`CUDA_VISIBLE_DEVICES=0`, SHALL record PyTorch CUDA availability, device count,
stderr, and import order, and SHALL rerun the selected-Python probe under a
sanitized CUDA environment that strips ROCm/XDNA path pollution without
mutating the parent process.

The ledger SHALL record `nvidia-smi` availability, NVIDIA driver version, CUDA
version, visible devices, and GPU memory state.  It SHALL record the tracked
environment variables `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`,
`CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `CMAKE_ARGS`, and `FORCE_CMAKE`, and
SHALL summarize ROCm, XDNA/Vitis/Xilinx, CUDA, Python path, and CMake pollution
findings as machine-readable entries.

When safe under the repository dependency workflow, Exp 3220 SHALL compare the
selected `.venv` against a temporary CUDA-only Python environment outside
protected source paths.  If the temporary environment is created, the artifact
SHALL record the exact creation/install/probe commands and CUDA package
versions.  If it is not created or cannot be created, the artifact SHALL record
the skip/failure as a repair-action entry instead of silently treating the
comparison as successful.

The artifact SHALL expose at least `schema_version`, `experiment_id`,
`milestone`, `selected_python`, `selected_python_cuda_ok_before`,
`selected_python_cuda_ok_after`, `isolated_cuda_venv_created`,
`isolated_cuda_venv_cuda_ok`, `cuda_visible_devices`,
`nvidia_smi_available`, `gpu_count_nvidia_smi`, `driver_version`,
`torch_version_selected`, `torch_cuda_version_selected`,
`environment_pollution_findings`, `repair_actions_attempted`,
`cuda_receipt_ready_candidate`, `recommended_next_action`,
`inference_substrate`, `conductor_file_modified`, `active_roadmap_modified`,
and `honest_verdict`.  `cuda_receipt_ready_candidate` SHALL be true only when a
clean selected-Python subprocess can initialize CUDA and identify at least one
GPU through PyTorch or an equivalent CUDA runtime probe.

### SCENARIO-INFER-SOTA-026-001: Sanitized Selected Python Identifies CUDA

**Given** `nvidia-smi` reports at least one NVIDIA GPU
**And** the selected Python probe with `CUDA_VISIBLE_DEVICES=0` reports
PyTorch CUDA available or an equivalent CUDA runtime device count greater than
zero after environment sanitization
**When** Exp 3220 builds the repair ledger
**Then** `selected_python_cuda_ok_after=true`
**And** `cuda_receipt_ready_candidate=true`
**And** the artifact keeps `inference_substrate` on a no-model CUDA-runtime
forensics value.

### SCENARIO-INFER-SOTA-026-002: Isolated CUDA Venv Separates Venv From System Boundary

**Given** the selected Python probe still cannot initialize CUDA
**And** a temporary CUDA-only Python environment is safely created outside the
repository source tree
**When** the isolated CUDA runtime probe identifies a GPU
**Then** `isolated_cuda_venv_created=true`
**And** `isolated_cuda_venv_cuda_ok=true`
**And** `recommended_next_action` identifies selected-venv CUDA repair rather
than system driver repair.

### SCENARIO-INFER-SOTA-026-003: Environment Pollution Is Explicit And Non-Successful

**Given** the tracked environment contains ROCm, XDNA/Vitis/Xilinx, Python path,
or CMake CUDA build variables
**When** Exp 3220 builds the repair ledger
**Then** `environment_pollution_findings` records each polluted variable class
**And** a successful `nvidia-smi` result alone does not set
`cuda_receipt_ready_candidate=true`
**And** the artifact never claims SOTA model receipt.

### REQ-INFER-SOTA-027: Exp 5097 Clean SOTA Endpoint Logprob Cache Runtime Provenance

The system SHALL provide an Exp 5097 runtime-provenance repair artifact at
`results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json` and, only
when real token logprob or top-logprob evidence is observed from a local
llama.cpp/OpenAI-compatible endpoint, a smoke cache at
`results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.jsonl`.

Exp 5097 SHALL inspect the three mandated GGUF IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through the shared local GGUF resolver /
`cached_sota_pair()` pattern and SHALL NOT call `AutoTokenizer` on a `*-GGUF`
repository ID.  Before endpoint inference or server bring-up, the artifact
SHALL record `preconditions_checked` with CUDA/GPU visibility, free VRAM when
available, llama.cpp Python/server availability, resolved local `.gguf` paths,
a free-port probe, cache path/disk-space evidence, and the runtime environment
variables used.

The artifact SHALL either probe an already-running endpoint or start a bounded
local `llama-server` process for one resolved mandated GGUF.  It SHALL record
server command/PID when a process is started, endpoint URL, start/end timing,
endpoint lifetime, server logs, cleanup behavior, deterministic completion
proof, token logprob or top-logprob proof, cache row count/path, and an
adversarial verification result before declaring readiness.  Exp 5097 SHALL
write at least ten smoke cache rows only when real token logprob or top-logprob
evidence is present; otherwise it SHALL set `cache_rows_written=0` and record
`blocker_root_cause`.

The artifact SHALL expose these required fields with matching entries in
`field_principles`: `honest_verdict`, `duration_s`, `inference_substrate`,
`preconditions_checked`, `model_specs`, `usable_sota_models`, `server_command`,
`endpoint_url`, `endpoint_lifetime_s`, `completion_endpoint_ready`,
`logprob_endpoint_ready`, `top_logprob_or_confidence_ready`,
`sample_completion`, `sample_logprob_evidence`, `cache_rows_written`,
`cache_path`, `logprob_endpoint_clean`, `live_llm_invoked`,
`adversarial_verify_passed`, `blocker_root_cause`, and
`flagged_adversarial`.  `honest_verdict` SHALL use an honest terminal-prefix
form such as `success_clean_sota_endpoint_logprob_cache_ready` for a clean
endpoint/cache artifact or
`blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs` when local
logprob evidence is unavailable.

The required `field_principles` SHALL include these principle texts:
`terminal-prefix verdict separates a clean endpoint/cache result from an honest blocked runtime-precondition result.`;
`wall-clock duration catches live-substrate claims that are too short to be credible.`;
`declares live_llm_inference only when a local endpoint actually returned completion/logprob evidence.`;
`records CUDA/GPU, VRAM, llama.cpp, GGUF cache, free-port, cache-path, disk, and env evidence before inference.`;
`names all three mandated SOTA GGUF IDs with resolved local .gguf paths or missing diagnostics.`;
`the exact mandated local GGUF files that can be handed to llama.cpp.`;
`exact command proves whether a local server was started instead of silently assuming one existed.`;
`concrete endpoint URL makes the completion/logprob proof replayable.`;
`measured endpoint lifetime prevents a too-short live-substrate readiness claim.`;
`true only when a local endpoint returned non-empty deterministic completion text.`;
`true only when real token logprob or top-logprob evidence was observed.`;
`true only when top-logprob alternatives or structured confidence evidence are present.`;
`stores the deterministic completion proof used for readiness.`;
`stores observed token/top-logprob counts and examples without fabricating absent telemetry.`;
`cache rows are counted only when real endpoint telemetry backs them.`;
`stable JSONL path for the smoke cache or blocked no-row evidence.`;
`true only after endpoint, cache, and adversarial provenance checks all pass.`;
`true only when the local LLM endpoint returned completion/logprob evidence during this run.`;
`records the repository adversarial verifier outcome before readiness is declared.`;
`machine-readable missing endpoint, server, runtime, cache, or telemetry evidence for blocked runs.`;
and `true if the artifact's own provenance or adversarial verifier detects an inconsistent claim.`

### SCENARIO-INFER-SOTA-027-SUCCESS: Live Logprobs Produce Clean Cache

**Given** at least one mandated GGUF resolves to a local `.gguf` file
**And** a local endpoint returns deterministic completion text plus token
logprob or top-logprob evidence
**When** Exp 5097 clears the live-substrate duration/provenance checks
**Then** `completion_endpoint_ready=true`
**And** `logprob_endpoint_ready=true`
**And** `top_logprob_or_confidence_ready=true`
**And** `cache_rows_written>=10`
**And** `adversarial_verify_passed=true`
**And** `flagged_adversarial=false`
**And** the verdict is
`success_clean_sota_endpoint_logprob_cache_ready`.

### SCENARIO-INFER-SOTA-027-BLOCKED: Missing Live Logprobs Writes Clean Blocked Artifact

**Given** the mandated GGUF cache and endpoint preconditions are recorded
**But** no local endpoint returns real token logprob or top-logprob evidence
**When** Exp 5097 completes
**Then** `live_llm_invoked=false`
**And** `cache_rows_written=0`
**And** `inference_substrate=precondition_check_only`
**And** `blocker_root_cause` names the missing endpoint, server, runtime, or
telemetry evidence
**And** the artifact does not claim uPRM, process-verifier, or hallucination
detection wins.

### REQ-INFER-SOTA-028: Exp 5119 SOTA Endpoint Root-Cause Artifact

The system SHALL provide an Exp 5119 local SOTA endpoint root-cause artifact at
`results/experiment_5119_sota_endpoint_rootcause_v469.json`.  The artifact SHALL
diagnose local GGUF completion/logprob readiness before any downstream
benchmark, and SHALL use only `cached_sota_pair()` / `resolve_cached_gguf()` for
`*-GGUF` model resolution.  It SHALL NOT call `AutoTokenizer` on a GGUF
repository ID.

Exp 5119 SHALL name the three mandated `MODEL_SPECS` entries
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, recording exact local `.gguf` paths or
machine-readable blockers.  Before endpoint inference or server launch, it
SHALL record CUDA/GPU visibility, llama.cpp / llama-cpp-python availability,
disk/RAM, model cache evidence, and port availability.

If a local server is launched, Exp 5119 SHALL record the command, PID, port,
startup log, request/response transcript, endpoint lifetime, shutdown behavior,
and errors.  It SHALL attempt a minimal completion smoke and a logprob or
top-logprob smoke through the intended local backend.  Cache readiness SHALL be
true only when real token logprob or top-logprob evidence is observed; otherwise
`cache_ready=false` and no cache rows are promoted.

The artifact SHALL include at least these fields: `experiment_id`,
`milestone`, `honest_verdict`, `inference_substrate`, `duration_s`,
`preconditions_checked`, `MODEL_SPECS`, `cached_sota_pair_attempted`,
`gguf_paths`, `cuda_status`, `server_command`, `endpoint_lifetime_s`,
`completion_proof`, `logprob_proof`, `cache_ready`, `root_cause_tree`,
`flagged_adversarial`, and `tests_run`.  `experiment_id` SHALL equal
`exp5119-sota-endpoint-rootcause-v469`, and `milestone` SHALL equal
`2026.07.469`.

The `root_cause_tree` SHALL distinguish exact blocker keys
`missing_binary`, `wrong_model_path`, `unsupported_logprob_api`,
`cuda_failure`, `oom`, `timeout`, and `cache_schema_mismatch`.
The artifact SHALL run repository adversarial verification after writing its
JSON and SHALL mark `flagged_adversarial=true` if a critical verifier finding or
too-short live-model claim is detected.

### SCENARIO-INFER-SOTA-028-SUCCESS: Live Completion And Logprobs Prove Readiness

**Given** at least one mandated GGUF resolves to a local `.gguf` file
**And** the intended local backend returns deterministic completion text plus
token logprob or top-logprob evidence
**When** Exp 5119 clears adversarial verification and the live-model duration
floor
**Then** `cache_ready=true`
**And** `completion_proof.ready=true`
**And** `logprob_proof.ready=true`
**And** `flagged_adversarial=false`
**And** the verdict begins with `success_` or `complete_`.

### SCENARIO-INFER-SOTA-028-BLOCKED: Missing Live Logprobs Produces Root Cause

**Given** the mandated GGUF cache and compute preconditions are recorded
**But** no local backend returns real token logprob or top-logprob evidence
**When** Exp 5119 completes
**Then** `cache_ready=false`
**And** `inference_substrate` does not claim clean live logprob readiness
**And** `root_cause_tree` records the applicable blocker class
**And** the verdict begins with `blocked_`.

### REQ-INFER-SOTA-029: Exp 5124 Clean SOTA Runtime Provenance Gate

The system SHALL provide an Exp 5124 runtime-provenance artifact at
`results/experiment_5124_clean_sota_runtime_provenance_v470.json` that proves
or blocks the local endpoint/cache/logprob substrate before downstream LLM
experiments run.  Exp 5124 SHALL resolve `MODEL_SPECS` by calling
`cached_sota_pair()` first, SHALL record the exact `model_path` values returned
for the mandated local GGUFs, and SHALL use only llama.cpp / GGUF paths for
runtime checks.  It SHALL NOT call `AutoTokenizer` on `*-GGUF` repositories.

Exp 5124 SHALL attempt a real nontrivial local completion through an
already-running llama.cpp-compatible endpoint or by starting a bounded local
`llama-server` for one resolved mandated GGUF.  The artifact SHALL record
`completion_proof`, `logprob_proof`, `endpoint_lifetime_s`,
`request_response_transcript`, `cache_receipts`, and cache write/read evidence.
It SHALL record real wall-clock `duration_s`; it SHALL NOT mock or pad timing.
For clean live claims it SHALL include `duration_floor_evidence` explaining why
repository adversarial verification should not raise `DURATION_TOO_SHORT`.

The artifact SHALL expose these fields: `experiment_id`, `milestone`,
`honest_verdict`, `inference_substrate`, `duration_s`, `MODEL_SPECS`,
`cached_sota_pair_attempted`, `gguf_paths`, `completion_proof`,
`logprob_proof`, `cache_ready`, `cache_receipts`, `endpoint_lifetime_s`,
`request_response_transcript`, `duration_floor_evidence`,
`adversarial_verify_passed`, `sota_runtime_clean`, `conductor_modified`, and
`tests_run`.  `experiment_id` SHALL equal
`exp5124-clean-sota-runtime-provenance-v470`, `milestone` SHALL equal
`2026.07.470`, `inference_substrate` SHALL equal
`local_sota_gguf_llamacpp_runtime_or_blocked`, and `conductor_modified` SHALL
be false.

`sota_runtime_clean` SHALL be true only when completion evidence, logprob
evidence, cache write/read evidence, and adversarial verification all pass.  If
any precondition fails, Exp 5124 SHALL write a terminal blocked artifact with
`sota_runtime_clean=false`, `cache_ready=false`, an honest `blocked_` verdict,
and a machine-readable `root_cause_tree`.

### SCENARIO-INFER-SOTA-029-CLEAN: Live Completion Logprobs Cache And Verifier Pass

**Given** `cached_sota_pair()` returns two mandated local GGUF model specs with
`model_path` values
**And** the intended local llama.cpp-compatible backend returns non-empty
completion text plus token logprob or top-logprob evidence
**And** the smoke cache writes and reads back at least one row
**When** Exp 5124 records real wall-clock duration and adversarial verification
returns no critical flags
**Then** `completion_proof.ready=true`
**And** `logprob_proof.ready=true`
**And** `cache_ready=true`
**And** `adversarial_verify_passed=true`
**And** `sota_runtime_clean=true`
**And** the verdict begins with `success_` or `complete_`.

### SCENARIO-INFER-SOTA-029-BLOCKED: Missing Runtime Evidence Blocks Downstream Gate

**Given** the mandated GGUF cache resolution attempt is recorded
**But** completion, logprob, cache, or adversarial verification evidence is
missing or failed
**When** Exp 5124 completes
**Then** `sota_runtime_clean=false`
**And** `cache_ready=false`
**And** `root_cause_tree` records the failed precondition
**And** the verdict begins with `blocked_`
**And** downstream verifier experiments are not run by this task.

### REQ-INFER-SOTA-030: Exp 5125 Structured Reasoning Candidate Pool

The system SHALL provide an Exp 5125 non-FoVer structured-reasoning candidate
pool at `results/experiment_5125_structured_reasoning_pool_v470.json` with pool
rows written to `results/experiment_5125_structured_reasoning_pool_v470.jsonl`.
Exp 5125 SHALL load
`results/experiment_5124_clean_sota_runtime_provenance_v470.json` and SHALL
hard-block when `sota_runtime_clean` is not true.  When the gate is open, the
experiment SHALL build 80-150 exact-checkable tasks across at least three
families, such as small constraint-satisfaction tasks, Knights-and-Knaves logic,
travel/budget constraints, and code/property checks.

Exp 5125 SHALL define `MODEL_SPECS` through `cached_sota_pair()` when possible,
SHALL include at least one mandated local GGUF model path, and SHALL record
`model_path` provenance for any candidate generation attributed to an LLM.  It
SHALL NOT use FoVer data or a FoVer selector scope.  Candidate ground truth SHALL
come only from deterministic validators; no LLM judge SHALL be treated as the
oracle.  The experiment SHALL compute `oracle_at_k`, `cheap_baseline_at_1`,
per-family headroom, `parse_coverage`, `duplicate_rate`, and copy-rate evidence.

The terminal artifact SHALL expose these fields: `experiment_id`, `milestone`,
`honest_verdict`, `inference_substrate`, `duration_s`, `MODEL_SPECS`,
`task_families`, `pool_path`, `pool_sha256`, `pool_n`, `candidates_per_item`,
`exact_validators_used`, `oracle_at_k`, `cheap_baseline_at_1`,
`parse_coverage`, `duplicate_rate`, `structured_pool_ready`,
`verifier_is_oracle`, `fover_scope_used`, `conductor_modified`, and
`tests_run`.  `experiment_id` SHALL equal
`exp5125-structured-reasoning-pool-v470`, `milestone` SHALL equal
`2026.07.470`, `inference_substrate` SHALL equal
`local_sota_gguf_generation_with_exact_validators`, `verifier_is_oracle` SHALL
be false, `fover_scope_used` SHALL be false, and `conductor_modified` SHALL be
false.

`structured_pool_ready` SHALL be true only when `pool_n` is between 80 and 150,
`parse_coverage` clears the declared usability gate, and
`oracle_at_k - cheap_baseline_at_1` clears the declared headroom gate.  If the
Exp 5124 gate is closed or no mandated local GGUF path can be recorded, Exp 5125
SHALL write a terminal blocked artifact with `structured_pool_ready=false` and
SHALL NOT fabricate candidate rows.

### SCENARIO-INFER-SOTA-030-POOL: Exact Validators Produce Headroom

**Given** Exp 5124 reports `sota_runtime_clean=true`
**And** at least one mandated local GGUF model path is resolved
**When** Exp 5125 builds the structured reasoning pool
**Then** the pool contains 80-150 non-FoVer tasks across at least three
families
**And** deterministic validators score every parseable candidate
**And** `oracle_at_k` exceeds `cheap_baseline_at_1`
**And** `structured_pool_ready=true` only when the pool-size, parse-coverage,
and headroom gates all pass.

### SCENARIO-INFER-SOTA-030-BLOCKED: Dirty Runtime Gate Blocks Candidate Rows

**Given** Exp 5124 is missing or reports `sota_runtime_clean=false`
**When** Exp 5125 runs
**Then** it writes a terminal blocked artifact
**And** `structured_pool_ready=false`
**And** `pool_n=0`
**And** no FoVer scope or LLM judge is used as ground truth.

### REQ-INFER-SOTA-033: Exp 5137 Solver-Verified Formulation Selector

The system SHALL provide an Exp 5137 non-FoVer solver-verified formulation
generation and selection evaluation at
`results/experiment_5137_solver_verified_formulation_selector_v471.json`, with
problem-model-code records written to
`results/experiment_5137_solver_verified_formulation_selector_v471_pmc.jsonl`.
Exp 5137 SHALL load
`results/experiment_5136_receipt_structured_pool_v2_v471.json`, SHALL hard-block
unless `structured_pool_v2_clean=true`, and SHALL use the Exp 5136 pool rows
only as non-FoVer exact-checkable OR/CSP task provenance.

When the upstream gate is open, Exp 5137 SHALL evaluate a bounded subset of
small exact-checkable OR/CSP tasks where multiple mathematical formulations are
plausible. It SHALL generate structured problem-model-code records for the
mandated local GGUF model IDs `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; SHALL verify each formulation through a
solver backend plus exact post-checks; and SHALL report feasibility restoration
separately from original model formulation quality.

Exp 5137 SHALL compare the exact-feedback formulation selector against static
hand formulation, seeded random valid formulation, cheapest-feasible repair,
and direct-answer baselines. It SHALL report selector delta versus the strongest
static or cheap baseline, CI95, feasibility rate, wrong-label count, solve
effort delta, and family-holdout behavior. `formulation_selector_ready` SHALL be
true only when the selector improves over the strongest static or cheap baseline
with zero wrong labels and passing controls.

The terminal artifact SHALL expose these fields: `experiment_id`, `milestone`,
`honest_verdict`, `inference_substrate`, `duration_s`, `MODEL_SPECS`,
`upstream_pool_artifact`, `formulation_families`, `pmc_records_path`,
`solver_backend`, `feasibility_restoration_used`,
`selector_delta_vs_best_static`, `delta_ci95`, `wrong_label_count`,
`solve_effort_delta`, `formulation_selector_ready`, `fover_scope_used`,
`conductor_modified`, and `tests_run`. `experiment_id` SHALL equal
`exp5137-solver-verified-formulation-selector-v471`, `milestone` SHALL equal
`2026.07.471`, and `inference_substrate` SHALL equal
`local_sota_gguf_formulation_generation_with_solver_verification`.

### SCENARIO-INFER-SOTA-033-SELECTOR: Solver Feedback Selects Verified Formulations

**Given** Exp 5136 reports `structured_pool_v2_clean=true`
**And** the Exp 5136 pool JSONL is loadable
**When** Exp 5137 evaluates solver-verified formulation records
**Then** every selected formulation passes an exact post-check
**And** feasibility restoration is reported separately from original
formulation quality
**And** selector utility is compared against static hand, random-valid,
cheapest-feasible repair, and direct-answer baselines
**And** `formulation_selector_ready=true` only if the selector beats the
strongest static or cheap baseline with zero wrong labels and passing controls.

### SCENARIO-INFER-SOTA-033-BLOCKED: Dirty Structured Pool Gate Fails Closed

**Given** Exp 5136 is missing or reports `structured_pool_v2_clean=false`
**When** Exp 5137 runs
**Then** it writes a terminal blocked artifact
**And** `formulation_selector_ready=false`
**And** `pmc_records_path` is null
**And** no solver-verified formulation records are fabricated.

### REQ-INFER-SOTA-032: Exp 5136 Receipt-Backed Structured Reasoning Pool V2

The system SHALL provide an Exp 5136 receipt-backed non-FoVer
structured-reasoning pool v2 at
`results/experiment_5136_receipt_structured_pool_v2_v471.json` with pool rows
written to `results/experiment_5136_receipt_structured_pool_v2_v471.jsonl`.
Exp 5136 SHALL load both
`results/experiment_5124_clean_sota_runtime_provenance_v470.json` and
`results/experiment_5125_structured_reasoning_pool_v470.json`, SHALL
hard-block when Exp 5124 does not report clean local SOTA runtime provenance,
and SHALL hard-block any attempted path that depends on FoVer data, FoVer
selector scope, or an LLM judge as ground truth.

When the gates are open, Exp 5136 SHALL build 100-160 exact-checkable
structured tasks across at least three non-FoVer families, SHALL define
`MODEL_SPECS` with all three mandated local GGUF model IDs
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, and SHALL prefer `cached_sota_pair()` while
recording exact model paths for all three mandated models. Candidate correctness
SHALL be scored only by deterministic exact validators.

For every generated candidate batch, Exp 5136 SHALL emit a receipt record with
the prompt hash, model specification, endpoint or command provenance,
wall-clock start and stop timestamps, raw response hash, parsed candidate hash,
validator output hash, and candidate correctness result. The terminal artifact
SHALL expose these fields: `experiment_id`, `milestone`, `honest_verdict`,
`inference_substrate`, `duration_s`, `MODEL_SPECS`, `preconditions_checked`,
`receipt_records`, `duration_floor_evidence`, `task_families`, `pool_path`,
`pool_sha256`, `pool_n`, `candidates_per_item`, `exact_validators_used`,
`oracle_at_k`, `cheap_baseline_at_1`, `parse_coverage`, `duplicate_rate`,
`family_headroom`, `structured_pool_v2_clean`, `adversarial_verify_passed`,
`verifier_is_oracle`, `fover_scope_used`, `conductor_modified`, and
`tests_run`. `experiment_id` SHALL equal
`exp5136-receipt-structured-pool-v2-v471`, `milestone` SHALL equal
`2026.07.471`, and `inference_substrate` SHALL equal
`local_sota_gguf_generation_with_receipts_and_exact_validators`.

`structured_pool_v2_clean` SHALL be true only when the pool size, parse
coverage, oracle headroom, mandated-model provenance, receipt completeness,
duration-floor evidence, and adversarial verification gates all pass.

### SCENARIO-INFER-SOTA-032-POOL: Receipts Back Every Exact Candidate

**Given** Exp 5124 reports clean local SOTA GGUF runtime provenance
**And** no upstream or current path uses FoVer scope
**When** Exp 5136 builds the structured pool v2
**Then** the pool contains 100-160 exact-checkable tasks across at least three
families
**And** `MODEL_SPECS` records all three mandated local GGUF model IDs with
model paths
**And** every candidate has a matching receipt record containing prompt,
response, parsed-candidate, validator-output, model, command or endpoint, and
wall-clock provenance hashes
**And** `oracle_at_k` exceeds `cheap_baseline_at_1`
**And** deterministic exact validators are the only ground truth.

### SCENARIO-INFER-SOTA-032-BLOCKED: Dirty Or FoVer Preconditions Fail Closed

**Given** Exp 5124 is missing or not clean
**Or** Exp 5125 or the current run declares FoVer scope
**When** Exp 5136 runs
**Then** it writes a terminal blocked artifact
**And** `structured_pool_v2_clean=false`
**And** `pool_n=0`
**And** no candidate rows are fabricated.

### REQ-INFER-SOTA-031: Exp 5126 Distributional Energy Ranker

The system SHALL provide an Exp 5126 CPU ranker/abstainer over the Exp 5125
structured-reasoning pool at
`results/experiment_5126_distributional_energy_ranker_v470.json`.  Exp 5126
SHALL load `results/experiment_5125_structured_reasoning_pool_v470.json`, SHALL
hard-block when `structured_pool_ready` is not true, and SHALL use
`results/experiment_5125_structured_reasoning_pool_v470.jsonl` as the only
candidate pool.

The ranker SHALL decompose energy into deterministic exact-checkable constraint
penalties, a learned or calibrated quality score, and an ensemble uncertainty
term used for abstention.  Deterministic validators SHALL remain the ground
truth; FoVer data and LLM judges SHALL NOT be used as correctness oracles.  The
train/calibration/test split SHALL keep all candidates for a task item in the
same split and SHALL stratify by task family so duplicate item leakage cannot
inflate headline metrics.

Exp 5126 SHALL compare the distributional-energy ranker against cheap
non-learned baselines, including length/features, parse validity,
constraint-count-only, and seeded random among parse-valid candidates.  It SHALL
report accuracy@1, violation rate, abstention rate, AUROC or ranking metrics
where applicable, CI95 for the paired delta versus the strongest cheap baseline,
family-holdout results, label-shuffle, duplicate, and model-identity shortcut
controls.  `ranker_ready_for_audit` SHALL be true only when the headline CI95
excludes zero and the controls pass.

The terminal artifact SHALL expose these fields: `experiment_id`, `milestone`,
`honest_verdict`, `inference_substrate`, `duration_s`, `source_pool_path`,
`MODEL_SPECS`, `deterministic_constraint_penalties`,
`learned_quality_score_description`, `uncertainty_abstention_rule`,
`strongest_cheap_baseline`, `distributional_energy_delta`, `delta_ci95`,
`family_holdout_results`, `label_shuffle_result`,
`model_identity_shortcut_check`, `ranker_ready_for_audit`,
`verifier_is_oracle`, `conductor_modified`, and `tests_run`.
`experiment_id` SHALL equal `exp5126-distributional-energy-ranker-v470`,
`milestone` SHALL equal `2026.07.470`, `inference_substrate` SHALL equal
`cpu_ranker_over_exact_validated_sota_candidates`, `verifier_is_oracle` SHALL be
false, and `conductor_modified` SHALL be false.

### SCENARIO-INFER-SOTA-031-RANKER: Decomposed Ranker Reports Honest Gate

**Given** Exp 5125 reports `structured_pool_ready=true`
**And** the structured pool JSONL is loadable
**When** Exp 5126 fits and evaluates the CPU distributional-energy ranker
**Then** every split is grouped by task item and stratified by family
**And** deterministic constraint penalties, learned quality scoring, and
uncertainty abstention are reported separately
**And** the headline delta is computed against the strongest cheap baseline
**And** `ranker_ready_for_audit=true` only if CI95 excludes zero and leakage
controls pass.

### SCENARIO-INFER-SOTA-031-BLOCKED: Closed Structured Pool Gate Blocks Ranker

**Given** Exp 5125 is missing or reports `structured_pool_ready=false`
**When** Exp 5126 runs
**Then** it writes a terminal blocked artifact
**And** `ranker_ready_for_audit=false`
**And** no learned or baseline metric is fabricated from candidate rows.

### REQ-INFER-018: EBT Abstraction Layer

The system SHALL provide an Energy-Based Transformer (EBT) abstraction layer that bridges autoregressive LLMs (like the mandated SOTA GGUF models) to an EBT formulation. This enables System-2 gradient refinement at inference time by providing a way to calculate sequence energy.
- `ebt_bridge.py` SHALL define an `EBTBridge` class that takes a model and calculates `sequence_energy` from logits or logprobs.
- The `sequence_energy` SHALL be formulated such that higher probability sequences have lower energy.

### SCENARIO-INFER-018-001: Sequence Energy Calculation

**Given** an initialized `EBTBridge` wrapping an autoregressive model
**When** a text sequence is passed to the bridge
**Then** it returns a valid sequence energy value
**And** sequences with higher likelihood are assigned lower energy

### REQ-INFER-1829: EqM Calibration
The system SHALL provide an EqM calibration implementation in `python/carnot/inference/eqm_calibration.py` that computes adaptive computation steps for stable valley finding as described in arXiv:2510.02300.

### REQ-INFER-2041: EqM Gradient Extraction on FAR Continuous Space
The system SHALL provide an EqM gradient extraction module in `python/carnot/inference/far_eqm.py` that maps continuous hidden states (e.g., from `unsloth/gemma-4-26B-A4B-it-GGUF`) to an energy landscape gradient. It MUST support optimization-driven refinement using EqM and be evaluated against toy constraints.

### SCENARIO-INFER-2041-001: Optimization-driven refinement
**Given** continuous hidden states mapped to an energy landscape
**When** the EqM optimization step is applied across 10 toy constraints
**Then** the energy landscape gradient strictly reduces the total energy
**And** the experiment validation results are saved to `results/experiment_2041_eqm_gradients.json`.

### REQ-INFER-2056: Soft Bellman ARM-to-EBM Solver

The system SHALL provide a Rust `soft_bellman_solve` implementation that maps
autoregressive token log-probabilities into an EBM-compatible sequence energy.
For normalized ARM log-probabilities, the solver SHALL use the canonical
probability gauge of the soft Bellman equation: terminal soft value is zero,
all prefix soft values solve to zero, immediate rewards equal token log
probabilities, and token energies are their negation.  The solver SHALL return
per-token rewards, per-token energies, solved soft values, sequence affinity,
sequence energy, sequence log-probability, and the root log-partition value.

The PyO3 layer SHALL expose the solver to Python as `soft_bellman_solve`.
The implementation SHALL reject non-finite or positive token log-probabilities
instead of silently producing invalid EBM energies.

### SCENARIO-INFER-2056-001: ARM Logprobs Produce EBM Energies

**Given** a finite sequence of ARM token log-probabilities
**When** `soft_bellman_solve` is called
**Then** each immediate reward equals the corresponding log-probability
**And** each token energy equals `-logprob`
**And** the sequence energy equals the negative sum of token log-probabilities
**And** the returned soft values satisfy the canonical zero-value Bellman gauge.

### SCENARIO-INFER-2056-002: Exhaustive Sequence Energies Normalize

**Given** all token paths from a small normalized ARM distribution
**When** each path is converted with `soft_bellman_solve`
**Then** `exp(-sequence_energy)` for all paths sums to one
**And** lower ARM likelihood paths receive higher EBM energy.

### REQ-INFER-SOTA-3327: Exp 3327 Energy Descent Substrate Bootstrap Smoke

The system SHALL provide a bootstrap smoke test for energy-descent versus autoregressive generation using local SOTA GGUF inference in `scripts/experiment_3327_energy_descent_substrate_bootstrap_v1.py`.
The script SHALL write `results/experiment_3327_energy_descent_substrate_bootstrap_v1.json` and decide whether `exp3328` may run.
The script SHALL use `cached_sota_pair(gpu_indices=(0, 1))` to require at least one of `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF`.
It SHALL preflight cache paths, GPU visibility, and runtime loader health, writing a blocked artifact if preconditions fail.
It SHALL run a tiny deterministic smoke of 8-16 arithmetic or constraint prompts, recording energy trajectory, verifier score trajectory, candidate text fingerprints, model_specs, batch timing, and whether any exact verifier score improves.
It SHALL NOT claim a scientific delta, and SHALL set `energy_descent_bootstrap_ready=true` only if live inference, telemetry, and duration/provenance checks are clean.

### SCENARIO-INFER-SOTA-3327-001: Bootstrap Smoke Opens Gate

**Given** SOTA GGUF cache is available and GPU preconditions are met
**When** the energy-descent bootstrap smoke runs
**Then** it runs a tiny deterministic smoke and writes telemetry
**And** the artifact sets `energy_descent_bootstrap_ready=true` only if all checks pass.

### REQ-INFER-3414: FR-11 Adversarial Robustness Stress Test
The system SHALL execute a full stress test of FR-11 integrating the new NUP phase transition and Latent Spills signals.
It SHALL initialize the VerifyRepairPipeline with FR-11 metrics enabled.
It SHALL process a dataset combining normal and adversarially perturbed CoT traces using `MODEL_SPECS = ["unsloth/Qwen3.6-35B-A3B-GGUF"]`.
It SHALL evaluate final system accuracy and self-calibration limits and output `results/experiment_3414_fr11_adversarial_robustness.json`.

### SCENARIO-INFER-3414-001: FR-11 Adversarial Robustness Execution
**Given** FR-11 metrics enabled in VerifyRepairPipeline
**When** adversarially perturbed CoT traces are processed with Qwen3.6
**Then** it evaluates accuracy and self-calibration limits
**And** the experiment artifact `results/experiment_3414_fr11_adversarial_robustness.json` is generated.

### REQ-INFER-3406: Latent Spills Sensing EBM Pipeline
The system SHALL provide an EBM pipeline to read latent activation energies and detect 'spills' signaling guessing.
The verification pipeline SHALL be instrumented to extract internal latent states using `MODEL_SPECS = ["unsloth/gemma-4-31B-it-GGUF"]`.
The EBM SHALL calculate energy spills dynamically per token.
The pipeline SHALL validate against a small hallucination dataset and output an artifact `results/experiment_3406_latent_spills_sensing.json` with all required schema fields including `honest_verdict` and `spill_detection_ready`.

### SCENARIO-INFER-3406-001: Latent Spills Sensing Pipeline Execution
**Given** the model unsloth/gemma-4-31B-it-GGUF and a small hallucination dataset
**When** the latent spills sensing EBM pipeline is executed
**Then** it calculates energy spills dynamically per token
**And** the experiment artifact `results/experiment_3406_latent_spills_sensing.json` is generated.

## Implementation Status

| Requirement | Python | Tests |
|------------|--------|-------|
| REQ-INFER-001 | Implemented | 15+ Python |
| REQ-INFER-002 | Implemented | 8+ Python |
| REQ-INFER-003 | Implemented | 8+ Python |
| REQ-INFER-004 | Implemented | 6+ Python |
| REQ-INFER-005 | Implemented | 7+ Python |
| REQ-INFER-006 | Implemented | 14+ Python |
| REQ-INFER-007 | Implemented | 18+ Python |
| REQ-INFER-014 | Implemented | 35 Python |
| REQ-INFER-015 | Implemented | 14 Python |
| REQ-INFER-016 | Implemented | 10 Python |
| REQ-INFER-017 | Implemented | 7 Python |
| REQ-INFER-SOTA-004 | Implemented | 4 Python |
| REQ-INFER-SOTA-005 | Implemented | 4 Python |
| REQ-INFER-SOTA-006 | Implemented | 12 Python |
| REQ-INFER-SOTA-007 | Implemented | 18+ Python |
| REQ-INFER-SOTA-008 | Implemented | 12 Python |
| REQ-INFER-SOTA-009 | Implemented | 7 Python |
| REQ-INFER-SOTA-010 | Implemented (`python/carnot/reporting/live_sota_balanced_telemetry_v2.py`) | Implemented (`tests/python/test_experiment_1480_live_sota_balanced_telemetry_v2.py`) |
| REQ-INFER-SOTA-011 | Proposed | Proposed |
| REQ-INFER-SOTA-012 | Implemented (`python/carnot/reporting/sota_runtime_preflight.py`) | Implemented (`tests/python/test_experiment_2836_sota_runtime_preflight.py`) |
| REQ-INFER-SOTA-013 | Implemented (`python/carnot/reporting/sota_runtime_cache_offload_resolver_v3.py`) | Implemented (`tests/python/test_experiment_2862_sota_runtime_cache_offload_resolver_v3.py`) |
| REQ-INFER-SOTA-014 | Planned | Planned |
| REQ-INFER-SOTA-016 | Implemented (`python/carnot/reporting/sota_energy_micro_panel_logprob_corrigendum_v2.py`) | Implemented (`tests/python/test_experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.py`) |
| REQ-INFER-SOTA-017 | Implemented (`python/carnot/reporting/sota_micro_panel_clean_telemetry_v3.py`) | Implemented (`tests/python/test_experiment_2886_sota_micro_panel_clean_telemetry_v3.py`) |
| REQ-INFER-SOTA-018 | Implemented (`python/carnot/reporting/spilled_energy_logit_detector_micro_panel_v1.py`) | Implemented (`tests/python/test_experiment_2917_spilled_energy_logit_detector_micro_panel.py`) |
| REQ-INFER-SOTA-021 | Implemented (`scripts/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.py`) | Implemented (`tests/python/test_experiment_3013_sota_gguf_logprob_telemetry_preflight.py`) |
| REQ-INFER-SOTA-022 | Implemented (`python/carnot/experiment_3043_verified_speculation_transcript_fingerprint.py`) | Implemented (`tests/python/test_experiment_3043_verified_speculation_transcript_fingerprint.py`) |
| REQ-INFER-SOTA-023 | Implemented (`python/carnot/reporting/sota_cache_preconditions_manifest_v2.py`) | Implemented (`tests/python/test_experiment_3123_sota_cache_preconditions_manifest_v2.py`) |
| REQ-INFER-SOTA-024 | Planned (`python/carnot/reporting/cuda_env_forensics_ledger_3206.py`) | Planned (`tests/python/test_experiment_3206_cuda_env_forensics_ledger.py`) |
| REQ-INFER-SOTA-025 | Implemented (`python/carnot/reporting/llama_cpp_cuda_rebuild_clean_subprocess_3207.py`) | Implemented (`tests/python/test_experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess.py`) |
| REQ-INFER-SOTA-026 | Implemented (`python/carnot/reporting/hermetic_cuda_runtime_repair_ledger_3220.py`) | Implemented (`tests/python/test_experiment_3220_hermetic_cuda_runtime_repair_ledger.py`) |
| REQ-INFER-SOTA-027 | Planned (`python/carnot/experiment_5097_clean_sota_endpoint_logprob_cache.py`, `results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json`) | Planned (`tests/python/test_experiment_5097_clean_sota_endpoint_logprob_cache.py`) |
| REQ-INFER-SOTA-028 | Implemented (`python/carnot/experiment_5119_sota_endpoint_rootcause.py`, `scripts/experiment_5119_sota_endpoint_rootcause_v469.py`, `results/experiment_5119_sota_endpoint_rootcause_v469.json`) | Implemented (`tests/python/test_experiment_5119_sota_endpoint_rootcause.py`) |
| REQ-INFER-SOTA-029 | Planned (`python/carnot/experiment_5124_clean_sota_runtime_provenance.py`, `scripts/experiment_5124_clean_sota_runtime_provenance_v470.py`, `results/experiment_5124_clean_sota_runtime_provenance_v470.json`) | Planned (`tests/python/test_experiment_5124_clean_sota_runtime_provenance.py`) |
| REQ-INFER-SOTA-030 | Implemented (`python/carnot/experiment_5125_structured_reasoning_pool_v470.py`, `scripts/experiment_5125_structured_reasoning_pool_v470.py`, `results/experiment_5125_structured_reasoning_pool_v470.json`) | Implemented (`tests/python/test_experiment_5125_structured_reasoning_pool_v470.py`) |
| REQ-INFER-SOTA-031 | Implemented (`python/carnot/experiment_5126_distributional_energy_ranker_v470.py`, `scripts/experiment_5126_distributional_energy_ranker_v470.py`, `results/experiment_5126_distributional_energy_ranker_v470.json`) | Implemented (`tests/python/test_experiment_5126_distributional_energy_ranker_v470.py`) |
| REQ-INFER-SOTA-032 | Implemented (`python/carnot/experiment_5136_receipt_structured_pool_v2_v471.py`, `scripts/experiment_5136_receipt_structured_pool_v2_v471.py`, `results/experiment_5136_receipt_structured_pool_v2_v471.json`) | Implemented (`tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py`) |
| REQ-INFER-SOTA-033 | Implemented (`python/carnot/experiment_5137_solver_verified_formulation_selector_v471.py`, `scripts/experiment_5137_solver_verified_formulation_selector_v471.py`, `results/experiment_5137_solver_verified_formulation_selector_v471.json`) | Implemented (`tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py`) |
| REQ-INFER-2041 | Proposed | Proposed |
| REQ-INFER-2056 | Implemented (`crates/carnot-boltzmann/src/lib.rs`, `crates/carnot-python/src/lib.rs`) | Implemented (`crates/carnot-boltzmann/tests/soft_bellman.rs`, `tests/python/test_soft_bellman_pyo3.py`) |

### REQ-INFER-1957: TruncProof Token-Limited LL(1) Parsing

The system SHALL provide a budget-aware LL(1) grammar parser in `python/carnot/inference/grammar.py` that implements a TruncProof-style budget-aware closure. The parser SHALL track a token budget. When the token budget nears exhaustion, the parser SHALL automatically force deterministic closing tokens to seal the structure to avoid structurally invalid JSON.
The zero-false-accept parsing rate on truncated runs MUST be evaluated.
The parser SHALL output an artifact `results/experiment_1957_truncproof_ll1_grammar.json`.

### SCENARIO-INFER-1957-001: Budget-Aware Closure Seals Structure

**Given** an LL(1) parser with a token budget of 5
**When** 4 tokens have been consumed
**Then** the parser forces the closing token
**And** the resulting structure is valid

### REQ-INFER-1970: DOMINO-Style Speculative Grammar Masking

The system SHALL provide a DOMINO-style speculative grammar masker (`DominoGrammarMasker`) in `python/carnot/inference/grammar.py` that builds a subword-aligned transition mask generator for common Carnot schema patterns. It SHALL integrate with the decoding loop to enforce strict adherence without latency spikes, and evaluate exact-acceptance equivalence against a legacy trie path.
The masker SHALL output an artifact `results/experiment_1970_domino_fast_constraints.json` that includes exact-acceptance equivalence and latency comparisons.

### SCENARIO-INFER-1970-001: DOMINO Masking Evaluates Exact Acceptance

**Given** a DOMINO-style grammar masker initialized with Carnot schema patterns
**When** it evaluates token acceptance vs a legacy trie path
**Then** exact-acceptance equivalence is confirmed
**And** the experiment artifact `results/experiment_1970_domino_fast_constraints.json` is written

### REQ-INFER-1958: GCoT-Decoding Reasoning Paths
The system shall provide a GCoTBranchingSampler that maintains parallel latent reasoning traces, applies partial-trace energy to rank and cull branches, and implements backtracking when all active branches exceed the energy threshold.

### REQ-INFER-SOTA-2009: Dual Model VRAM Allocator Test for ROCm
The system SHALL provide a multi-model VRAM allocator test for ROCm in `scripts/experiment_2009_dual.py`.
The allocator SHALL mock VRAM allocation for SOTA models `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` if hardware is absent.
The allocator SHALL determine theoretical max sequence length based on total VRAM and model weights.
The output SHALL be saved to `results/experiment_2009_dual_model.json` with all required schema fields including `honest_verdict` and calculated `max_sequence_length`.

### SCENARIO-INFER-SOTA-2009-001: Dual Model VRAM Allocation
**Given** the mandated models Qwen 35B and Gemma 31B
**When** the dual model VRAM allocator is executed
**Then** theoretical max sequence length is determined
**And** the result is saved to `results/experiment_2009_dual_model.json`.



### REQ-INFER-CRANE-2089: Augmented Grammar Decoder
The inference engine SHALL implement an Augmented Grammar Decoder in `python/carnot/inference/crane_decoder.py`.
The decoder SHALL apply standard logits until a specific trigger token is generated, and then strictly apply a BNF grammar constraint.
The output SHALL be saved to `results/experiment_2089_crane_decoder.json` with `crane_ready=true`.

### SCENARIO-INFER-CRANE-2089-001: Augmented Grammar Decoder Execution
**Given** the models unsloth/gemma-4-31B-it-GGUF and unsloth/Qwen3.6-35B-A3B-GGUF
**When** the CRANE decoder is executed
**Then** it interleaves constrained and unconstrained decoding
**And** the result is saved to `results/experiment_2089_crane_decoder.json`.

### REQ-INFER-CRANE-2090: CRANE HumanEval Evaluation
The system SHALL evaluate CRANE vs rigid start-to-finish grammar enforcement on 50 HumanEval problems requiring structured JSON output in `scripts/experiment_2090_crane_humaneval.py`.
The experiment SHALL use model `unsloth/gemma-4-31B-it-GGUF`.
The output SHALL be saved to `results/experiment_2090_crane_humaneval.json` with `pass_rate_delta > 0` and include fields `target`, `pipeline_invocations`, `simulated_energy_minimized`, `latency_ms`, `honest_verdict`, `crane_pass_rate`, `rigid_pass_rate`.

### SCENARIO-INFER-CRANE-2090-001: CRANE HumanEval Run
**Given** 50 HumanEval problems
**When** evaluated with CRANE versus rigid grammar
**Then** CRANE achieves a positive pass_rate_delta
**And** the artifact `results/experiment_2090_crane_humaneval.json` is generated.

### REQ-INFER-2133: Hardware-Assisted DAB
The system SHALL provide a Hardware-Assisted DAB logits processor in `python/carnot/inference/hw_dab.py`.
It SHALL offload energy evaluations to a simulated LUT representation based on Substrate-Aware KANs.
The output SHALL be saved to `results/experiment_2133_hw_dab.json` with all required schema fields including `hw_dab_ready`.

### SCENARIO-INFER-2133-001: Hardware-Assisted DAB Evaluation
**Given** an LLM outputting logits
**When** the HWDABLogitsProcessor is applied
**Then** it updates logits by subtracting energy from simulated LUTs
**And** the experiment artifact `results/experiment_2133_hw_dab.json` is generated.

### REQ-INFER-3412: VGS Textual Constraint Decoding
The system SHALL adapt the VGS penalty logic to use explicit textual constraint grounding.
An active decoding modifier (LogitsProcessor) SHALL penalize autoregressive probabilities that disagree with explicit constraints.
The output SHALL be saved to `results/experiment_3412_vgs_override_decoder.json` showing hallucination avoidance statistics.

### SCENARIO-INFER-3412-001: VGS Textual Constraint Decoder Application
**Given** an LLM generating text with explicit textual constraints
**When** evaluated with VGSTextualConstraintLogitsProcessor
**Then** probabilities disagreeing with constraints are penalized
**And** hallucination avoidance statistics are reported in `results/experiment_3412_vgs_override_decoder.json`.
