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
