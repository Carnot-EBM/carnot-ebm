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
