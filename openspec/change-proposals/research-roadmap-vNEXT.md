# Research Roadmap vNEXT: Milestone 2026.07.480

Status: proposed
Milestone title: Typed Memory, Receipt Integrity, and Verified Decoding Allocation

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

## What 2026.07.479 Proved

Milestone `.479` closed a useful but uneven research loop:

- The controlled memory loop produced a real positive result: aligned memory beat shuffled and no-memory controls, rollback was exercised, retention passed, and no model training claim was made.
- The KAN certificate path moved beyond the tiny single-axis case: the certificate now covers 10 PWA segments across two variables, includes false-property rejection, and still avoids broad hardware or KAN generality claims.
- ARC rubric-to-patch synthesis produced a provenance-routed live patch, but the follow-up live run had `level_delta=0` and was methodology-flagged. No new ARC level should be banked from `.479`.
- GAP-4 and GAP-1 remain blocked by receipt/methodology instability, not by a clean negative scientific result.
- The current VerIbmc local solver-feedback route is retired: the methodology-correct rerun showed `solver_feedback_uplift=0.0`.
- Hardware continuity stayed honest: KV260 and PolarFire remain reachable; GateMate remains blocked by physical/JTAG setup; no speedup was claimed.

## Three Biggest Gaps to the PRD Vision

1. Artifact credibility is still the first bottleneck.

   The PRD asks for verifiable reasoning with auditable evidence, but recent milestones repeatedly lost candidate wins to missing receipts, malformed gate fields, duration flags, and post-hoc artifact ambiguity. The next milestone must make producer-side artifact validation a first-class research object before reusing borderline results.

2. Continuous self-learning is promising but not yet transferable.

   `.479` proved that aligned memory can matter inside a controlled setup. The PRD vision is broader: persistent learning that improves future verifier decisions without hidden training or leakage. The next step is cross-model, typed memory transfer between the mandated local SOTA GGUF models, plus a verifier-dose scheduler that decides when memory is enough and when full verification is required.

3. Live-path verification remains too narrow.

   ARC still has no new live level, hallucination verification has not been stress-tested in multi-turn citation settings, and KAN certificates are still toy-scale. The next milestone must improve live-agent evidence paths, add a hallucination/provenance microbench, and test a stronger convex-envelope KAN certificate without reviving retired external text-scorer routes.

## 2025-2026 Research Incorporated

The following findings were added to `research-references.md` before this roadmap was designed:

- MemCollab, arXiv 2603.23234: cross-model memory via contrastive trajectory distillation and task-aware retrieval. This motivates the cross-model typed-memory experiment.
- HalluHard, arXiv 2602.01031: multi-turn hallucination benchmark with inline citation requirements and strong residual hallucination rates. This motivates a small provenance-memory hallucination microbench.
- Hybrid Verified Decoding, arXiv 2606.01019: dynamic choice between cache and model drafting using predicted accepted length and payoff. This motivates verifier-dose scheduling.
- Token-Guard, arXiv 2601.21969: token-level self-checking and regeneration using latent hallucination-risk estimates. This motivates a local fragment-level self-checking pilot.
- SLOT, arXiv 2505.04016: model-agnostic structured-output postprocessing with schema/content fidelity. This motivates artifact schema normalization.
- Efficient Convexification of KANs, arXiv 2604.03871: convex hulls and envelopes for polynomial KANs. This motivates the next KAN certificate step.
- Extropic TSU/XTR-0 public updates: still relevant for energy-native hardware direction, but not yet a local reproducible substrate.
- Logical Intelligence Aleph/Kona updates: reinforce the energy-based verification target, but public materials do not expose enough internals to implement Kona compatibility claims.
- Semantic Scholar status: ARM-EBM has early citations; EBT citation discovery was rate-limited and should be retried, but no actionable dependent experiment should wait on it.

## Architecture Target

```text
                         recent 2025-2026 papers
                                   |
                                   v
                    exp5246 SOTA ingestion and deltas
                                   |
                                   v
  raw experiment artifacts -> exp5247 SLOT-style schema/receipt normalizer
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
       exp5248 GAP-4 receipt salvage       local SOTA GGUF workers
                    |                             |
                    |                             v
                    |              exp5249 cross-model typed memory
                    |                             |
                    |                             v
                    |              exp5250 verifier-dose scheduler
                    |                             |
                    +--------------+--------------+
                                   |
                                   v
           deterministic verifiers, provenance gates, rollback checks
                    |              |              |
                    v              v              v
         exp5251 token guard  exp5252 HalluHard  exp5253 ARC live receipts
                                                     |
                                                     v
                                live self-discovery registry, no duplicate solves

       exp5254 KAN convex certificate      exp5255 hardware continuity
                    |                                  |
                    +----------------+-----------------+
                                     v
                          exp5256 capstone synthesis
```

The target is not a new monolith. It is a tighter evidence pipeline:

- Normalize and validate artifacts before promoting claims.
- Convert controlled memory gains into cross-model typed-memory evidence.
- Allocate verifier work based on measured payoff instead of using every verifier everywhere.
- Keep ARC solve credit on the live agent path only.
- Continue KAN and hardware work under conservative certificate and no-speedup boundaries.

## Phase Plan

### Phase 0: Close and Refresh

- `exp5245`: archive `.479` into `research-complete.yaml`, `ops/status.md`, `ops/changelog.md`, and traceability.
- `exp5246`: refresh SOTA references against the V480 additions and rerun Semantic Scholar/OpenReview/HF/GitHub checks without using `/deep-research`.

### Phase 1: Evidence Integrity

- `exp5247`: implement a SLOT-inspired artifact schema and receipt normalizer outside `scripts/research_conductor.py`.
- `exp5248`: use the normalizer to salvage, reject, or retire the current GAP-4 pool. This is a receipts task, not a new generation sweep.

### Phase 2: Continuous Self-Learning and Allocation

- `exp5249`: run cross-model typed-memory transfer using the mandated local GGUF models. This is the milestone's required continuous self-learning experiment.
- `exp5250`: if cross-model memory is eligible, build a verifier-dose scheduler inspired by Hybrid Verified Decoding to decide when memory, cheap deterministic checks, or full local SOTA verification is warranted.
- `exp5251`: run a Token-Guard/Carnot fragment self-checking pilot with local SOTA GGUF generation and deterministic energy/provenance gates.
- `exp5252`: run a HalluHard-style multi-turn provenance-memory microbench with local SOTA GGUF models and citation-support checks.

### Phase 3: Live Paths, Certificates, Hardware, Synthesis

- `exp5253`: rerun the `.479` ARC provenance patch with clean live-path receipts and retire the patch if it repeats `level_delta=0`.
- `exp5254`: extend KAN certification using convex-envelope ideas from arXiv 2604.03871.
- `exp5255`: continue hardware continuity with KV260 and PolarFire receipts, GateMate physical/JTAG status, and p-kit boundary notes. No speedup claim is allowed.
- `exp5256`: capstone synthesis and next-roadmap recommendations.

## Dependency Graph

```text
exp5245 archive
   |
   v
exp5246 SOTA refresh
   |
   +--> exp5247 schema/receipt normalizer --> exp5248 GAP-4 receipt salvage
   |
   +--> exp5249 cross-model typed memory --> exp5250 verifier-dose scheduler
   |
   +--> exp5251 Token-Guard/Carnot pilot
   |
   +--> exp5252 HalluHard provenance microbench
   |
   +--> exp5253 ARC live patch receipt-clean rerun
   |
   +--> exp5254 KAN convex-envelope certificate
   |
   +--> exp5255 hardware continuity

all completed or honestly blocked --> exp5256 capstone
```

Structured conductor gates:

- `exp5248` is gated on `exp5247.artifact_normalizer_ready == true`.
- `exp5250` is gated on `exp5249.cross_model_memory_eligible == true`.

## Model and Inference Requirements

Any experiment that calls an LLM must declare `MODEL_SPECS` and include at least one mandated local SOTA GGUF model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy tiny models may be used only for CPU smoke tests. They cannot be headline-result models.

Required inference-substrate labels for `.480`:

- `live_llm_inference_local_gguf_sota`: real local GGUF calls with model id, quantization, llama-server/llama.cpp command, seed, prompt checksum, completion checksum, and wall-clock receipts.
- `cached_fixture_replay_no_llm`: deterministic replay of existing candidate artifacts; no generation or model-quality claim.
- `offline_deterministic_certificate_no_llm`: deterministic certificate or verifier computation; no LLM claim.
- `hardware_probe_no_speedup_claim`: board reachability and environment receipts only.
- `offline_arcade_live_agent_runtime_self_discovery_no_llm`: ARC live-agent runtime receipts where solve credit must come from live self-discovery, not outer-loop reverse engineering.

## Hardware Requirements

- Local NVIDIA GPU with enough memory for at least one mandated GGUF headline model. If unavailable, LLM experiments must write an honest precondition failure artifact rather than falling back to legacy tiny headline models.
- KV260 reachable via SSH only. Do not require host `/dev/mmcblk*`.
- PolarFire reachable for continuity probes only unless a terminal workload already exists.
- GateMate remains blocked by physical/JTAG setup; report status only unless the operator has changed hardware access.
- Extropic TSU/XTR-0 and Kona/Aleph public updates are reference material only. Do not claim execution, compatibility, or speedup without local reproducible receipts.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not use `/deep-research`.
- Do not rerun retired scopes unless the task includes a complete `prior_failures` block and changes the technique, prerequisite, or retirement rule.
- Do not revive the retired Phase D external generated-text/logprob scorer path.
- Do not headline offline ARC BFS, per-game calibration solves, or outer-loop reverse engineering.
- Do not duplicate an ARC solve already present in `ops/arc_solve_registry.yaml`.
- Do not make hardware speedup claims from reachability probes.

## Success Criteria

Milestone `.480` succeeds if it produces at least one of the following clean positives, plus clean retirement or blocking decisions for the rest:

- Cross-model typed memory transfers useful verifier state between mandated local SOTA GGUF models without leakage or rollback failure.
- The verifier-dose scheduler reduces unnecessary full verification calls while preserving decision quality on a controlled replay set.
- GAP-4 is either salvaged with complete receipts or explicitly retired as unsalvageable in its current form.
- The ARC provenance patch is either banked through clean live-path self-discovery evidence or retired after a clean repeated zero-delta result.
- The KAN convex-envelope certificate covers a larger or more expressive property while rejecting a false property.
- Hardware continuity remains current and honest, with no unsupported speedup claims.
