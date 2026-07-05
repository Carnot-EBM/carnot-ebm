# Research Roadmap vNEXT: Milestone 2026.07.483

Created: 2026-07-05
Milestone: 2026.07.483
Status: proposed
Milestone title: Claim-Level Verification, Memory Attribution, and Solver-Hardware Guidance

## Inputs Read

- `research-program.md`
- `_bmad/prd.md`
- `_bmad/architecture.md`
- `ops/status.md`
- `ops/changelog.md`
- `research-complete.yaml`
- `research-roadmap.yaml`
- `openspec/change-proposals/`
- `ops/conductor-log.md`
- `research-references.md`
- `research-hardware-wishlist.md`
- `CLAUDE.md`
- `CODEX.md`
- `scripts/experiment_template.py`
- `ops/exclusion_manifest.yaml`

## What 2026.07.482 Proved

Milestone `.482` completed the governed-memory and certificate tracks while exposing the next verifier and
hardware bottlenecks:

- SOTA GGUF telemetry receipts are available for the mandated local model set. The milestone proved runtime
  and telemetry harness readiness, not verifier quality.
- The internal/logit hallucination probe was harmful relative to the lexical baseline. Its artifact showed a
  negative delta, so `.483` must not rerun the same internal/logit quality path without a changed mechanism.
- The deterministic solver fixture was rebuilt successfully with valid baselines and counterexample coverage.
- SOTA solver-grounded extraction stayed blocked because local GGUF GPU/offload preconditions failed during
  the retry. The next LLM extraction attempt needs a separate runtime repair gate.
- Governed decision-history memory was positive: provenance, scope, stale-conflict handling, rollback, and
  unsafe-action rejection all passed on the bounded fixture.
- Memory-assisted verifier dosing was positive: it preserved always-full verifier quality, avoided most full
  verifier calls on the replay fixture, and introduced no unsafe false accepts.
- KAN PWA/MILP certificates scaled positively with false-property rejection intact.
- The solver-to-factor-graph boundary is usable only at tiny scale. No hardware acceleration claim exists.
- Hardware continuity remains blocked: KV260 and PolarFire were unreachable, GateMate remained physical/JTAG
  blocked, and no speedup claim was made.
- Evidence normalization and the capstone closed cleanly, with honest positives, a harmful internal-verifier
  result, one quarantined/blocked SOTA extraction path, and one hardware reachability block.

## Three Biggest Gaps to the PRD Vision

1. **Oracle-distinct verification:** Carnot still lacks a robust verifier signal that beats cheap lexical or
   format baselines. The next milestone moves from raw internal/logit probes to claim-level
   knowledge-thought coherence and compilable trace verification.

2. **Continuous self-learning attribution:** Governed memory can safely preserve decisions and reduce verifier
   calls, but it does not yet explain which memory operation failed or improved the loop. `.483` adds
   operation-stage attribution across extraction, update, routing, maintenance, and use.

3. **Solver/certificate/hardware bridge:** Carnot has KAN certificates and a tiny factor-graph boundary, but
   not a solver-authoritative sampler-guidance result or reachable board evidence. `.483` adds CPU p-bit/CDCL
   guidance and keeps hardware continuity honest.

## Research Incorporated for 2026.07.483

The `V483 Research Update - 2026-07-05` section in `research-references.md` drives the design:

- CheckRLM motivates claim extraction, knowledge checking, and minimal correction as a safer next verifier
  target than the failed internal/logit signal.
- VeryTrace motivates converting reasoning chains into a tiny compilable DSL with dependency links,
  executable expressions, and localized repair labels.
- Constrained-decoding safety and reliability work warns that syntactic validity is not semantic safety.
  Schema compliance must be measured separately from correctness and false accepts.
- HaluMem, Agent-Native Memory, and MemTrace motivate operation-level memory failure attribution rather than
  final-decision-only memory metrics.
- G-RRM motivates solver-authoritative neural guidance: hints are allowed only when the symbolic solver can
  overwrite and recover from bad hints.
- Probabilistic-bit guided CDCL and p-bit simulated annealing papers motivate a CPU p-bit/CDCL guidance
  benchmark before any hardware acceleration claim.
- EBM theory on distributional simplicity bias motivates a low-order-factor certificate curriculum before
  higher-order factor claims.
- Extropic TSU and Logical Intelligence Kona/Aleph material remains architecture context only. No execution,
  compatibility, or speedup claim is made without local receipts.

## Architecture Target

```text
                         research-references.md V483 update
                                      |
                                      v
                         exp5282 archive .482 / activate .483
                                      |
                                      v
                         exp5283 SOTA/source delta refresh
                                      |
                                      v
                         exp5284 SOTA runtime/offload repair
                              |                         |
                              |                         v
                              |       exp5286 claim-level coherence SOTA pilot
                              |                ^        |
                              |                |        v
                              |       exp5290 memory-assisted coherence dosing
                              |                ^
                              |                |
                 exp5285 deterministic coherence fixture

                 exp5287 compilable trace DSL fixture
                              |
                              v
                 exp5288 SOTA trace DSL extraction retry

                 exp5289 memory operation attribution
                              |
                              v
                 exp5290 memory-assisted coherence dosing

                 exp5291 low-order factor certificate curriculum
                              |
                              v
                 exp5292 p-bit/CDCL solver guidance
                              |
                              v
                 exp5293 hardware continuity reachability

            all completed, gated-skipped, harmful, null, or blocked -> exp5294 capstone
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Runtime Preconditions

**Goal:** close `.482` honestly, keep the reference set current, and isolate whether local SOTA GGUF
generation/offload is ready before any LLM-dependent quality task runs.

- `exp5282-archive-482-activate-483`: archive the `.482` positive/null/harmful/blocked split and confirm
  `.483` is pre-staged without activating it.
- `exp5283-sota-source-delta-v483`: run the execution-time literature/source delta and append only genuinely
  new actionable items to `research-references.md`.
- `exp5284-sota-runtime-offload-receipt-repair-v483`: repair or honestly block the local SOTA GGUF generation
  path that blocked Exp5274, producing the `sota_offload_ready` gate.

### Phase 1 - Claim-Level and Trace-Level Verification

**Goal:** replace the harmful internal/logit route with verifier signals grounded in claims, knowledge,
solver checks, and executable traces.

- `exp5285-knowledge-thought-coherence-fixture-v483`: build a deterministic CheckRLM-style fixture with claim
  extraction, evidence links, minimal correction labels, lexical baselines, and safety negatives.
- `exp5286-knowledge-thought-coherence-sota-pilot-v483`: gated on runtime/offload and fixture readiness; run
  mandated SOTA GGUF models and report claim-level coherence quality without using the failed internal/logit
  score as a headline result.
- `exp5287-compilable-trace-dsl-fixture-v483`: build a VeryTrace-style tiny DSL fixture from the solver
  fixture, with dependency links, executable expressions, and repair labels.
- `exp5288-sota-trace-dsl-extraction-gated-v483`: gated on runtime/offload and trace-fixture readiness; retry
  SOTA extraction with solver-authoritative validation, overwrite/recovery telemetry, and false-accept
  accounting.

### Phase 2 - Continuous Self-Learning and Memory Attribution

**Goal:** advance PRD FR-11 by measuring not just whether governed memory helps, but where it succeeds or
fails inside the self-learning loop.

- `exp5289-memory-operation-attribution-v483`: attribute memory outcomes across extraction, update, routing,
  maintenance, and use using Exp5275/Exp5276 governed artifacts.
- `exp5290-memory-assisted-coherence-dose-gated-v483`: gated on coherence fixture and attribution readiness;
  test whether governed memory can safely allocate claim/coherence checks while preserving unsafe-false-accept
  controls.

### Phase 3 - Certificates, Sampler Guidance, Hardware Continuity, and Capstone

**Goal:** advance the non-LLM EBM path while keeping the solver authoritative and hardware claims receipt
bound.

- `exp5291-low-order-factor-certificate-curriculum-v483`: test a low-order-first KAN/Ising certificate
  curriculum with false-property rejection intact.
- `exp5292-pbit-cdcl-factor-guidance-v483`: use p-bit/Ising-style assumptions to guide CDCL on tiny factor
  fixtures, recording conflicts saved, fallback use, and distribution gates with no hardware speedup claim.
- `exp5293-hardware-continuity-reachability-v483`: check KV260, PolarFire, and GateMate reachability only;
  record blocked reasons and refuse speedup claims.
- `exp5294-capstone-v483`: synthesize positives, nulls, harmful results, gated skips, hardware blocks, and
  retirement recommendations for the next milestone.

## Dependency Graph and Structured Gates

```text
exp5282 -> exp5283 -> exp5284

exp5285 -> exp5286 [gate: exp5284.sota_offload_ready == true
                    AND exp5285.coherence_fixture_ready == true]

exp5287 -> exp5288 [gate: exp5284.sota_offload_ready == true
                    AND exp5287.trace_dsl_ready == true]

exp5289 -> exp5290 [gate: exp5285.coherence_fixture_ready == true
                    AND exp5289.memory_attribution_ready == true]

exp5291 -> exp5292 -> exp5293

all terminal upstreams -> exp5294
```

Structured conductor gates:

- `exp5286` is gated on `exp5284.sota_offload_ready == true` and
  `exp5285.coherence_fixture_ready == true`.
- `exp5288` is gated on `exp5284.sota_offload_ready == true` and
  `exp5287.trace_dsl_ready == true`.
- `exp5290` is gated on `exp5285.coherence_fixture_ready == true` and
  `exp5289.memory_attribution_ready == true`.

## Model and Inference Requirements

Any experiment that calls an LLM must declare `MODEL_SPECS` and include at least one mandated local SOTA GGUF
model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy tiny models may be used only for CPU smoke tests. They cannot be headline-result models. New
experiment scripts must use the `cached_sota_pair()` pattern or the current repo SOTA helper and load GGUFs
through llama.cpp by local `.gguf` path. Do not use `AutoTokenizer` on a GGUF repository.

Required inference-substrate labels for `.483`:

- `literature_ingestion_network_sources`: network-backed source refresh only.
- `live_llm_inference_local_gguf_sota`: real local GGUF generation/scoring with model id, quantization,
  llama.cpp command, prompt checksum, output checksum, wall-clock receipt, and GPU/offload receipt.
- `offline_deterministic_fixture_no_llm`: fixture, parser, lexical baseline, schema, or safety-negative work
  with no live model-quality claim.
- `offline_deterministic_certificate_no_llm`: solver, MILP, KAN, factor graph, or CDCL computation with no
  LLM claim.
- `aggregation_from_upstream_artifacts`: capstone, replay, attribution, or scheduler aggregation with no live
  model-quality claim.
- `hardware_probe_no_speedup_claim`: board reachability and environment receipts only.

## Hardware Requirements

- Local NVIDIA GPUs are required for headline SOTA GGUF inference. If unavailable, LLM-dependent tasks must
  skip through gates or emit honest blocked artifacts.
- KV260 is reachable via SSH only. Do not require host `/dev/mmcblk*`.
- PolarFire remains reachability-only unless an authenticated terminal workload already exists.
- GateMate remains physical/JTAG blocked unless the operator changes the physical setup.
- Extropic TSU/XTR-0 and Logical Kona/Aleph are reference material only. Do not claim execution,
  compatibility, or speedup without local reproducible receipts.
- Hardware tasks must record `hardware_evidence_level`, `hardware_speedup_claimed`, `blocked_reason`, and the
  exact commands/probes run.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not use `/deep-research`.
- Do not rerun Exp5272's harmful internal/logit hallucination probe as a headline verifier signal.
- Do not revive the retired Phase D external generated-text/logprob scorer path.
- Do not treat syntactic/JSON/grammar validity as semantic correctness.
- Do not claim hardware speedups from reachability checks, paper references, public roadmap material, or
  CPU-only simulations.
