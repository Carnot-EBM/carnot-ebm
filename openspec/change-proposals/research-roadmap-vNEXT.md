# Research Roadmap vNEXT: Milestone 2026.07.482

Created: 2026-07-05
Milestone: 2026.07.482
Status: proposed
Milestone title: Receipt-Clean Internal Verification, Governed Self-Learning, and Hardware-Bound Certificates

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
- `ops/known-issues.md`

## What 2026.07.481 Proved

Milestone `.481` closed the local-runtime blocker and narrowed the next research step:

- SOTA GGUF runtime is now preflight-ready with local llama.cpp receipts. The usable headline models are
  `unsloth/gemma-4-31B-it-GGUF`, `unsloth/Qwen3.6-35B-A3B-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`; no quality claim was made by the preflight itself.
- Cross-model typed memory produced a clean null: no-memory, aligned-memory, and shuffled-memory controls
  tied on the bounded fixture, with no unsafe false accepts.
- Typed-memory policy checks were positive on cached replay: retention stayed at 1.0, interference stayed at
  0.0, stale-conflict eviction passed, and harmful-memory rollback passed.
- Solver-grounded constraint extraction was flagged/partial and scientifically weak: validity was worse than
  baseline on the small pilot, while unsafe false accepts stayed at zero.
- Internal neuron/logit-energy verification was flagged/partial: a weak logit-energy signal existed, but the
  artifact was quarantined by the duration/substrate gate and cannot be used as a headline claim.
- Verifier-dose scheduler replay was positive on cached evidence: decision quality was preserved, false
  accepts did not increase, and most full-verifier calls were avoided on the replay fixture.
- KAN certificate explanation/refinement stayed positive: explanation consistency and refinement behavior
  improved without weakening false-property rejection.
- Hardware thermodynamic schedule work stayed honestly blocked: KV260 and PolarFire were unreachable in that
  run, GateMate remained physical/JTAG blocked, and no speedup was claimed.
- Artifact normalizer producer adoption was positive: `ExperimentTemplate` can normalize shape-only artifacts,
  preserve bare gates, and reject missing evidence before adversarial verification.

## Three Biggest Gaps to the PRD Vision

1. **Verifier signal gap:** Carnot now has local SOTA GGUF runtime, but it still lacks a receipt-clean,
   oracle-distinct verifier signal that survives adversarial verification. The next milestone must split the
   flagged `.481` solver/internal pilots into receipts-first harnesses and gated quality reruns.

2. **Self-learning loop gap:** Memory governance is safer than before, but Carnot has not yet demonstrated a
   continuous self-learning loop that promotes decision history, scopes memory, evicts stale conflicts,
   rolls back harmful entries, and then safely changes verifier allocation.

3. **Certificate and hardware path gap:** KAN certificates and hardware-bound samplers are promising but still
   disconnected. Carnot needs a bounded path from extracted constraints to certificates, factor graphs, and
   hardware reachability receipts, while refusing speedup claims without real board/runtime evidence.

## Research Incorporated for 2026.07.482

The `.482` design incorporates the `V482 Research Update - 2026-07-05` added to `research-references.md`:

- Internal verification: arXiv:2604.06277, arXiv:2601.14210, and arXiv:2604.03524 motivate hidden/logit
  probes, but also warn that internal signals can be task/model-specific or absent. Result: run a telemetry
  harness before any internal-verifier quality claim.
- Constrained generation and solver grounding: TRIDENT (arXiv:2506.09701), globally constrained decoding
  work, and code-based prompt-constraint assurance motivate executable constraints and finite-state/solver
  fixtures. Result: rebuild the solver fixture before rerunning SOTA extraction.
- Continuous self-learning: MemoPilot (arXiv:2606.08656), memory governance, Portable Agent Memory,
  MemLineage, and memory-poisoning work motivate decision-history rows with provenance, scope, rollback, and
  unsafe-action controls. Result: the self-learning phase targets governed memory, not broad fine-tuning.
- KAN and certificate work: KAN PWA/MILP verification (arXiv:2602.06737) and runtime neural-certificate
  monitoring (arXiv:2507.11987) motivate scaling the existing KAN certificate path with explicit slack and
  dynamic spot checks.
- Hardware/sampling: probabilistic FPGA sampling and Extropic TSU/XTR-0 public material support the long
  hardware thesis, but only as boundary context without local execution receipts.
- Logical Intelligence Kona/Aleph posts reinforce constraint-satisfaction and formal-verification positioning,
  but they remain non-reproducible public baselines.

## Architecture Target

```text
                          research-references.md V482 update
                                      |
                                      v
                         exp5269 archive .481 / activate .482
                                      |
                                      v
                         exp5270 SOTA/source delta refresh
                                      |
                                      v
                    exp5271 local SOTA telemetry receipt harness
                         |                         |
                         |                         v
                         |             exp5272 internal hallucination probe
                         |
                         +----------------------+
                                                |
                                                v
                         exp5276 memory-assisted verifier-dose pilot
                                                ^
                                                |
                         exp5275 governed decision-history memory

       exp5273 solver fixture rebuild ---> exp5274 SOTA constraint extraction retry

       exp5277 KAN certificate scale ---> exp5278 factor-graph/sampler boundary
                                                |
                                                v
                                      exp5279 hardware continuity

       exp5280 artifact/evidence audit ------------------------------+
                                                                     |
 all completed, gated-skipped, or honestly blocked ------------------+--> exp5281 capstone
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Runtime Receipts

**Goal:** preserve `.481` results, refresh external references, and make the local SOTA telemetry substrate
usable before downstream quality experiments run.

- `exp5269-archive-481-activate-482`: archive `.481`, create `.482` status scaffolding, and record the
  previous milestone's positive/null/blocked split without touching `research-roadmap.yaml`.
- `exp5270-sota-source-delta-v482`: refresh source deltas, Semantic Scholar status, and mandated model cache
  state without making quality claims.
- `exp5271-sota-telemetry-receipt-harness-v482`: build a receipts-first local SOTA GGUF telemetry harness for
  logits/hidden-state availability, prompt checksums, duration, model ids, llama.cpp command, and GPU/offload
  receipts.

### Phase 1 - Receipt-Clean Verifier Signals

**Goal:** rerun only the `.481` flagged verifier directions after their substrate/fixture blockers are
addressed.

- `exp5272-internal-hallucination-probe-gated-v482`: gated on `exp5271.telemetry_harness_ready == true`; run
  a small internal/logit probe with lexical baselines, duration receipts, and no external text scorer.
- `exp5273-solver-fixture-rebuild-v482`: rebuild the deterministic solver-labeled extraction fixture with
  schema checks, counterexamples, and baseline validity.
- `exp5274-solver-constraint-extraction-retry-gated-v482`: gated on `exp5273.solver_fixture_ready == true`;
  rerun constraint extraction with mandated local SOTA GGUF models and solver-scored artifacts.

### Phase 2 - Governed Continuous Self-Learning

**Goal:** advance the PRD FR-11 loop with memory that is useful only when it is scoped, provenance-backed, and
rollback-safe.

- `exp5275-governed-decision-history-memory-v482`: convert verifier/memory lessons into persistent
  decision-history rows with scope, provenance, rejected revisions, outcome, conflict, and rollback fields.
- `exp5276-memory-assisted-verifier-dose-gated-v482`: gated on `exp5271` and `exp5275`; measure whether
  governed memory safely reduces full-verifier calls while preserving decision quality and unsafe-false-accept
  controls.

### Phase 3 - Certificates, Hardware Boundary, and Evidence Discipline

**Goal:** keep the non-LLM EBM path moving while preserving receipt discipline.

- `exp5277-kan-milp-certificate-scale-v482`: scale KAN certificate refinement toward multi-component PWA/MILP
  properties with explicit slack, solve time, and false-property rejection.
- `exp5278-constraint-factor-graph-boundary-v482`: map a small solver fixture into a factor-graph/Ising-style
  boundary artifact with autocorrelation and sampler-interface metrics, no hardware speedup claim.
- `exp5279-hardware-continuity-reachability-v482`: update KV260, PolarFire, and GateMate reachability/status
  receipts without host SD assumptions and without speedup claims.
- `exp5280-artifact-normalizer-evidence-audit-v482`: audit producer-side artifact normalization and required
  field discipline after `.481` adoption.
- `exp5281-capstone-v482`: synthesize the milestone with clean positives, nulls, gated skips, blocked
  hardware, and follow-up/retirement decisions.

## Dependency Graph and Structured Gates

```text
exp5269
  -> exp5270
      -> exp5271
          -> exp5272  [gate: telemetry_harness_ready == true]
          -> exp5276  [gate: telemetry_harness_ready == true AND exp5275 memory ready]

exp5273
  -> exp5274  [gate: solver_fixture_ready == true]

exp5275
  -> exp5276  [gate: memory_decision_history_ready == true]

exp5277
  -> exp5278
      -> exp5279

exp5280
  -> exp5281

all terminal upstreams -> exp5281
```

Structured conductor gates:

- `exp5272` is gated on `exp5271.telemetry_harness_ready == true`.
- `exp5274` is gated on `exp5273.solver_fixture_ready == true`.
- `exp5276` is gated on `exp5271.telemetry_harness_ready == true` and
  `exp5275.memory_decision_history_ready == true`.

## Model and Inference Requirements

Any experiment that calls an LLM must declare `MODEL_SPECS` and include at least one mandated local SOTA GGUF
model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy tiny models may be used only for CPU smoke tests. They cannot be headline-result models. New
experiment scripts should use the `cached_sota_pair()` pattern from `scripts/experiment_template.py` and
load GGUFs through llama.cpp by local `.gguf` path; do not use `AutoTokenizer` on a GGUF repository.

Required inference-substrate labels for `.482`:

- `live_llm_inference_local_gguf_sota`: real local GGUF generation or scoring with model id, quantization,
  llama.cpp command, prompt checksum, output checksum, wall-clock receipt, and GPU/offload receipt.
- `live_llm_internal_telemetry_local_gguf_sota`: local GGUF run that records which logits/hidden/attention
  fields are actually exposed, with duration and substrate receipts.
- `aggregation_from_upstream_artifacts`: cached replay, memory aggregation, scheduler replay, or capstone
  synthesis with no live model-quality claim.
- `offline_deterministic_certificate_no_llm`: solver, certificate, MILP, KAN, or factor-graph computation
  with no LLM claim.
- `hardware_probe_no_speedup_claim`: board reachability and environment receipts only.
- `literature_ingestion_network_sources`: network-backed literature/source refresh; no experiment outcome
  claim.

## Hardware Requirements

- Local NVIDIA GPUs are required for headline SOTA GGUF inference. If unavailable or if llama.cpp/GGUF GPU
  offload fails, LLM-dependent experiments must skip through structured gates or emit honest blocked
  artifacts; they must not fall back to tiny headline models.
- KV260 is reachable via SSH only. Do not require host `/dev/mmcblk*`.
- PolarFire remains a reachability/continuity target unless a terminal workload already exists.
- GateMate remains blocked by physical/JTAG setup unless the operator has changed the physical state.
- Extropic TSU/XTR-0 and Logical/Kona updates are reference material only. Do not claim execution,
  compatibility, or speedup without local reproducible receipts.
- Hardware tasks must record `hardware_evidence_level`, `hardware_speedup_claimed`, and `blocked_reason` when
  applicable.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not use `/deep-research`.
- Do not emit `agent_type: gemini`; active `CLAUDE.md` routing defaults experiment tasks to
  `agent_type: codex` and `model: gpt-5.5`.
- Do not rerun retired scopes unless the task includes a complete `prior_failures` block and a changed
  prerequisite, technique, or authorized override.
- Do not revive the retired Phase D external generated-text/logprob scorer path.
- Do not rerun the retired ARC provenance-routing patch or claim duplicate/off-path ARC solves.
- Do not make hardware speedup claims from reachability, public roadmap material, or CPU-only simulations.

## Success Criteria

Milestone `.482` succeeds if it produces one or more clean positives and honest terminal decisions for the
rest:

- The local SOTA telemetry substrate is either ready with receipts or cleanly blocked with actionable
  diagnostics.
- At least one verifier-signal rerun becomes receipt-clean, even if the scientific result is null.
- Continuous self-learning advances through governed decision-history memory and a memory-assisted verifier
  allocation pilot with unsafe-false-accept controls.
- Solver-grounded extraction improves over the `.481` flagged pilot or retires the narrow rerun scope with
  explicit evidence.
- KAN/factor-graph work extends certificate or sampler-boundary evidence without overclaiming hardware.
- Hardware documentation remains current and receipt-bound with no unsupported speedup claim.
- The capstone records which lines are clean positives, clean nulls, gated skips, or honest blocks, and names
  any scopes that should be retired if the same verdict recurs.
