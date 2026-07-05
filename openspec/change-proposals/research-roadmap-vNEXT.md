# Research Roadmap vNEXT: Milestone 2026.07.481

Status: proposed
Milestone title: Local SOTA Runtime, Internal Verification, and Self-Learning Memory Stability

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
- `ops/exclusion_manifest.yaml`

## What 2026.07.480 Proved

Milestone `.480` closed with useful negative and boundary-setting results:

- Artifact normalization is ready enough for downstream use, but the producer side still needs adoption so future artifacts do not depend on post-hoc cleanup.
- GAP-4 is no longer an ambiguous blocked win: the current pool was salvaged as a clean null with wins=0, losses=0, ties=120.
- Cross-model typed memory and verifier-dose allocation did not get measured because the local SOTA GGUF runtime hit a llama.cpp GPU-offload blocker.
- Token-Guard/Carnot fragment self-checking was harmful on its bounded pilot; the route should not be broadened without a different mechanism.
- HalluHard-style provenance memory was a clean null on the small fixture set; it did not reduce an already-zero unsupported-claim rate.
- ARC provenance patching produced `level_delta=0` with clean receipts and should be retired for the current patch route rather than rerun.
- KAN convex-envelope certification produced a bounded positive: a two-variable upper-bound certificate with false-threshold rejection and explicit slack.
- Hardware continuity stayed honest: KV260 and PolarFire remain reachable, GateMate remains blocked by physical/JTAG state, and no hardware speedup was claimed.

## Three Biggest Gaps to the PRD Vision

1. Local SOTA execution is still blocking headline evidence.

   The PRD requires verifiable reasoning with modern local models, and `CLAUDE.md` now mandates SOTA GGUF headline models. `.480` showed that memory transfer and verifier-dose experiments cannot be interpreted until the llama.cpp/GGUF runtime and GPU-offload path emit reproducible receipts or an honest precondition failure.

2. Continuous self-learning is not yet stable enough to trust.

   `.479` found a controlled typed-memory positive, while `.480` failed to measure cross-model transfer. The PRD's autonomous self-learning target requires more than one useful replay: Carnot needs retention, interference, promotion, eviction, rollback, and cross-model checks before memory can guide verifier decisions.

3. Oracle-distinct verification needs internal and solver-grounded routes.

   The Phase D external generated-text/logprob scorer is retired, Token-Guard/Carnot was harmful, and HalluHard was null. The next verifier frontier is not another text reranker; it is hidden/logit/attention energy signals, solver-grounded constraint extraction, deterministic verifier-dose replay, and tighter artifact production.

## 2025-2026 Research Incorporated

The following findings were added to `research-references.md` before this roadmap was designed:

- Satisfiability Solving with LLMs, arXiv 2605.28602: use SAT/CSP as a solver-grounded substrate for checking model-produced constraints.
- ConstrainPrompt / code-based assurance of prompt-defined constraints, OpenReview: motivate converting natural-language requirements into executable checks rather than external text scoring.
- Neuron-Level Evidence for Medical LLM Hallucination, arXiv 2607.00158: motivates internal-state hallucination probes.
- Detecting Contextual Hallucinations with Attention High-Frequency Energy, arXiv 2602.18145: motivates attention/logit-energy preflights when the runtime exposes the needed receipts.
- Modular Memory for Continual Learning Agents, arXiv 2603.01761, and When Continual Learning Moves to Memory, arXiv 2604.27003: motivate typed-memory interference and stability-plasticity audits.
- AgentOdyssey, arXiv 2606.24893, and ALMA, arXiv 2602.07755: future benchmarks and policy-search directions for test-time agent learning, after smaller typed-memory audits are clean.
- Scaling Up Thermodynamic AI Models, arXiv 2607.00170: motivates sampler-cost/autocorrelation hardware notes without claiming local thermodynamic execution.
- Extropic TSU/XTR-0 and Logical Intelligence Aleph/Kona public updates: support the long-term EBM/hardware direction but remain non-executable without local SDKs or reproducible internals.
- Semantic Scholar citation trails for EBT and ARM-EBM remain watch items; no citation changed the immediate `.481` priorities.

## Architecture Target

```text
                         V481 literature and status refresh
                                      |
                                      v
                exp5257 archive .480 -> exp5258 SOTA delta check
                                      |
                                      v
                     exp5259 local SOTA GGUF runtime preflight
                         |              |              |
             sota_runtime_ready     blocked       receipt-only
                         |
              +----------+-----------+
              |                      |
              v                      v
 exp5260 cross-model typed   exp5262 solver-grounded
 memory transfer retry       constraint extraction
              |                      |
              v                      v
       verifier memory        executable constraints,
       promotion/rollback     SAT/Z3 counterexamples

 exp5261 memory interference audit runs independently as the
 continuous-self-learning safety rail when live SOTA runtime is blocked.

              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
 exp5263 internal energy     exp5264 verifier-dose   exp5265 KAN certificate
 hallucination probe         scheduler replay        explanation/refinement

              +----------------------+----------------------+
                                      |
                                      v
                    exp5266 hardware thermodynamic boundary
                    exp5267 producer-side artifact adoption
                                      |
                                      v
                           exp5268 capstone synthesis
```

The target is a stricter evidence loop:

- Prove or block the mandated SOTA GGUF runtime before interpreting LLM-dependent experiments.
- Keep continuous self-learning alive through memory-stability audits even if live model execution is blocked.
- Replace retired external text-scoring with oracle-distinct internals, solver checks, and deterministic verifier allocation.
- Extend the bounded KAN certificate path without broad generality claims.
- Keep hardware work receipt-only unless real board/runtime execution changes.

## Phase Plan

### Phase 0: Closeout, SOTA Refresh, and Runtime Preflight

- `exp5257`: archive `.480`, update durable ops/research records, and prepare `.481` activation without modifying `research-roadmap.yaml`.
- `exp5258`: refresh SOTA references from the new V481 research block and append only genuinely new deltas.
- `exp5259`: unblock or honestly block the mandated local SOTA GGUF runtime using llama.cpp/GGUF receipts before any headline LLM task runs.

### Phase 1: Continuous Self-Learning and Memory Stability

- `exp5260`: gated on `exp5259.sota_runtime_ready == true`, retry cross-model typed-memory transfer with mandated local SOTA models.
- `exp5261`: independently audit typed-memory retention, interference, promotion, eviction, and rollback on deterministic fixtures. This is the milestone's always-runnable continuous self-learning task.

### Phase 2: Oracle-Distinct Verification

- `exp5262`: gated on SOTA runtime, run a solver-grounded constraint-extraction pilot inspired by SAT/CSP and ConstrainPrompt literature.
- `exp5263`: gated on SOTA runtime, test whether local internals/logits can support neuron/attention-energy hallucination signals; emit a clean blocked artifact if the runtime cannot expose them.
- `exp5264`: run a deterministic verifier-dose scheduler replay using cached fixtures, avoiding the live-model blocker that skipped `.480` exp5250.

### Phase 3: Certificates, Hardware, Evidence Production, and Synthesis

- `exp5265`: extend the KAN convex-envelope certificate with explanation/refinement and false-property rejection.
- `exp5266`: update hardware continuity with thermodynamic sampler-cost/autocorrelation boundaries from the 2026 literature; no speedup claim.
- `exp5267`: adopt the `.480` artifact normalizer at the producer/template boundary without changing `scripts/research_conductor.py`.
- `exp5268`: synthesize `.481`, retire or carry forward blocked scopes, and recommend the next milestone.

## Dependency Graph

```text
exp5257 archive .480
   |
   v
exp5258 SOTA refresh
   |
   +--> exp5259 SOTA GGUF runtime preflight
   |        |
   |        +--> exp5260 cross-model typed memory retry
   |        |
   |        +--> exp5262 solver-grounded constraint extraction
   |        |
   |        +--> exp5263 internal energy hallucination probe
   |
   +--> exp5261 typed-memory interference audit
   |
   +--> exp5264 verifier-dose scheduler replay
   |
   +--> exp5265 KAN certificate explanation/refinement
   |
   +--> exp5266 hardware thermodynamic boundary
   |
   +--> exp5267 artifact normalizer producer adoption

all completed, skipped by structured gate, or honestly blocked --> exp5268 capstone
```

Structured conductor gates:

- `exp5260` is gated on `exp5259.sota_runtime_ready == true`.
- `exp5262` is gated on `exp5259.sota_runtime_ready == true`.
- `exp5263` is gated on `exp5259.sota_runtime_ready == true`.

## Model and Inference Requirements

Any experiment that calls an LLM must declare `MODEL_SPECS` and include at least one mandated local SOTA GGUF model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy tiny models may be used only for CPU smoke tests. They cannot be headline-result models.

Required inference-substrate labels for `.481`:

- `live_llm_inference_local_gguf_sota`: real local GGUF calls with model id, quantization, llama-server or llama.cpp command, seed, prompt checksum, completion checksum, wall-clock receipt, and GPU/offload receipt.
- `llama_cpp_runtime_preflight_no_quality_claim`: GGUF loading/tokenization/offload preflight only; no model-quality claim.
- `cached_fixture_replay_no_llm`: deterministic replay of existing artifacts or fixtures; no generation or model-quality claim.
- `offline_deterministic_certificate_no_llm`: deterministic certificate, solver, or verifier computation; no LLM claim.
- `hardware_probe_no_speedup_claim`: board reachability and environment receipts only.
- `literature_ingestion_network_sources`: network-backed literature/source refresh; no experiment outcome claim.

## Hardware Requirements

- Local NVIDIA GPUs are required for headline SOTA GGUF inference. If unavailable or if llama.cpp/GGUF GPU offload fails, LLM-dependent experiments must skip through structured gates or emit honest blocked artifacts; they must not fall back to tiny headline models.
- KV260 is reachable via SSH only. Do not require host `/dev/mmcblk*`.
- PolarFire remains a reachability/continuity target unless a terminal workload already exists.
- GateMate remains blocked by physical/JTAG setup unless the operator has changed the physical state.
- Extropic TSU/XTR-0 and Logical/Kona updates are reference material only. Do not claim execution, compatibility, or speedup without local reproducible receipts.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not use `/deep-research`.
- Do not emit `agent_type: gemini`; active `CLAUDE.md` routing defaults to `agent_type: codex` and `model: gpt-5.5`.
- Do not rerun retired scopes unless the task includes a complete `prior_failures` block and a changed prerequisite, technique, or authorized override.
- Do not revive the retired Phase D external generated-text/logprob scorer path.
- Do not rerun the retired ARC provenance-routing patch or claim duplicate/off-path ARC solves.
- Do not make hardware speedup claims from reachability or public roadmap material.

## Success Criteria

Milestone `.481` succeeds if it produces one or more clean positives and honest terminal decisions for the rest:

- The mandated SOTA GGUF runtime is either unblocked with receipts or cleanly blocked with actionable diagnostics.
- Typed memory shows cross-model transfer or, at minimum, a clean interference/retention policy that advances continuous self-learning safely.
- Solver-grounded constraints or internal energy probes produce oracle-distinct verification signal without reopening retired external scorer paths.
- The verifier-dose scheduler replay defines a safe allocation policy from cached evidence.
- The KAN certificate path gains explanation/refinement coverage while still rejecting false properties.
- Hardware documentation remains current and receipt-bound with no unsupported speedup claim.
