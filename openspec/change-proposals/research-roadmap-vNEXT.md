# Research Roadmap vNEXT: Milestone 2026.07.484

Created: 2026-07-06
Milestone: 2026.07.484
Status: proposed
Milestone title: Changed SOTA Runtime, Adaptive Memory, and Solver-Gated Energy Stability

## Inputs Read

- `CLAUDE.md`
- `CODEX.md`
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
- `scripts/experiment_template.py`
- `ops/exclusion_manifest.yaml`

## What 2026.07.483 Proved

Milestone `.483` separated usable deterministic infrastructure from blocked live SOTA execution:

- The mandated SOTA GGUF path remains blocked for headline work. Exp5284 resolved model files and smoke paths
  but produced no GPU-offload evidence, so Exp5286 and Exp5288 were correctly gate-skipped.
- CheckRLM-style knowledge-thought coherence is now a usable deterministic fixture. Exp5285 produced seven
  labeled cases, caught unsafe false accepts, and separated lexical baseline weakness from semantic checking.
- VeryTrace-style compilable trace verification is now a usable deterministic fixture. Exp5287 built trace DSL
  cases with dependency links, executable checks, malformed controls, semantic-error controls, and repair labels.
- Continuous self-learning advanced on bounded fixtures. Exp5289 attributed memory-stage failures across all
  seven cases with unsafe propagation zero; Exp5290 preserved always-full quality while avoiding four of seven
  full claim/coherence checks.
- The low-order KAN/Ising curriculum was a clean null. Exp5291 certified every stage, but low-order ordering did
  not improve success over shuffled ordering.
- p-bit/CDCL guidance was mixed. Exp5292 saved aggregate conflicts on CPU simulation while harming the
  misleading-assumption class; correctness was preserved through solver fallback, and no hardware speedup was
  claimed.
- Hardware continuity stayed reachability-only. KV260 SSH was blocked, PolarFire provided status-only SSH
  reachability, GateMate remained physical/JTAG blocked, and no speedup claim was made.
- The `.483` operational retrospective had a timing-accounting mismatch. Planning should use artifacts,
  changelog, and conductor logs for research truth, not the locked zero-duration timing fields.

## Three Biggest Gaps to the PRD Vision

1. **Live local SOTA reasoning is still unavailable.** The PRD requires local-first open-weight verifier loops,
   but `.483` proved the current llama-cpp-python path is CPU-only for the mandated models. `.484` must change
   runtime substrate before asking for SOTA quality.

2. **Self-learning is useful but still fixed-policy.** Governed memory can reduce verifier calls safely on a
   tiny fixture, yet it does not adapt verifier-dose policy under held-out conflict, stale evidence, selective
   forgetting, or long-range memory pressure.

3. **Solver/energy components lack class gates and stability receipts.** LNS-style repair, p-bit/CDCL guidance,
   KAN abstraction, and EBT inner-loop control are promising, but `.483` showed distribution sensitivity and a
   null curriculum. The next milestone must add instance-class gates, spectral step controls, and dynamic
   certificate spot-checks before hardware or quality claims.

## Research Incorporated for 2026.07.484

The `V484 Research Update - 2026-07-06` section in `research-references.md` drives this plan:

- Official `llama-cpp-python`, vLLM GGUF, and Hugging Face GGUF docs make the runtime conclusion explicit:
  repeat CPU-only llama-cpp-python offload is retired; the next SOTA task must use a changed substrate or block.
- MemoryAgentBench motivates adaptive self-learning evaluation across accurate retrieval, test-time learning,
  long-range understanding, and conflict resolution rather than final decision quality only.
- The structured-output control-plane attack paper requires safety-negative controls for any trace DSL,
  grammar-constrained extractor, or structured SOTA output.
- V483 execution refresh items carry forward: ConsFormer-LNS motivates destroy/repair telemetry, AS2 motivates
  declarative constraint-group metadata with symbolic validation, and the EBT spectral-control artifact
  motivates lambda-max and step-size logging before inner energy descent is trusted.

Prior references remain active but are not re-promoted as new findings: Distributional EBMs, CheckRLM,
VeryTrace, HaluMem, MemTrace, G-RRM, p-bit/CDCL, Extropic TSU, Logical Intelligence Kona/Aleph, and KAN
PWA/MILP verification.

## Architecture Target

```text
                         V484 source update + .483 closeout
                                      |
                                      v
                         exp5295 archive .483 / activate .484
                                      |
                                      v
                         exp5296 bounded source delta refresh
                                      |
                                      v
                         exp5297 changed SOTA runtime substrate gate
                                      |
                         [gate: changed_runtime_sota_ready]
                                      |
                                      v
                         exp5298 SOTA coherence/trace smoke

       exp5285 coherence fixture -----------------------------^
       exp5287 trace DSL fixture ------------------------------^

       exp5299 constraint-LNS solver repair fixture
                    |
                    v
       exp5300 p-bit/CDCL instance-class gate

       exp5301 EBT spectral step-control diagnostic

       exp5290 fixed memory dose + MemoryAgentBench references
                    |
                    v
       exp5302 adaptive self-learning memory policy
                    |
                    v
       exp5303 stale/conflict/selective-forgetting memory stress

       exp5291 KAN null -> exp5304 dynamic abstraction spot-check
       exp5293 hardware block -> exp5305 reachability continuity

          all completed, gated-skipped, null, mixed, or blocked -> exp5306 capstone
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Changed Runtime Gate

**Goal:** close `.483` truthfully, keep references current, and avoid another doomed SOTA rerun through the
retired CPU-only path.

- `exp5295-archive-483-activate-484`: archive `.483` into durable records and prepare `.484` without
  overwriting `research-roadmap.yaml`.
- `exp5296-sota-source-delta-v484`: perform a bounded execution-time source refresh and append only genuinely
  new actionable items.
- `exp5297-changed-runtime-sota-substrate-gate-v484`: try a changed SOTA GGUF runtime substrate, such as
  native llama.cpp CUDA CLI/server, a fresh CUDA-enabled wheel/container, or a clearly documented alternate
  GGUF backend. Emit `changed_runtime_sota_ready`.
- `exp5298-sota-coherence-trace-smoke-gated-v484`: gated on Exp5297; run a tiny mandatory-model smoke over the
  `.483` coherence and trace fixtures, with no headline claim if runtime is still blocked.

### Phase 1 - Solver-Gated Repair and Energy Stability

**Goal:** advance non-LLM verification and sampler guidance while keeping symbolic solvers authoritative.

- `exp5299-constraint-lns-solver-repair-fixture-v484`: implement a deterministic ConsFormer-LNS-style
  destroy/repair fixture with classical baselines, solver validation, and safety-negative structured-output
  controls.
- `exp5300-pbit-cdcl-instance-class-gate-v484`: turn Exp5292's mixed result into an instance-class gate that
  blocks misleading-assumption classes and preserves aggregate guidance only where it helps.
- `exp5301-ebt-spectral-step-control-diagnostic-v484`: add a tiny EBT/energy-descent diagnostic that logs
  lambda-max estimates, step sizes, divergence, and recovery behavior before any energy-guided decoding claim.

### Phase 2 - Continuous Self-Learning

**Goal:** satisfy PRD FR-11 with adaptive, held-out, governed memory rather than a fixed replay policy.

- `exp5302-adaptive-memory-policy-self-learning-v484`: search/select verifier-dose memory policy on held-out
  deterministic cases, measuring call avoidance, quality preservation, unsafe false accepts, stale conflicts,
  and rollback.
- `exp5303-memory-stress-conflict-forgetting-v484`: stress the selected policy with MemoryAgentBench-style
  accurate retrieval, test-time learning, long-range understanding, conflict resolution, selective forgetting,
  and harmful-memory rollback controls.

### Phase 3 - Certificates, Hardware Continuity, and Capstone

**Goal:** turn null/mixed certificate and hardware findings into better-gated next evidence.

- `exp5304-kan-dynamic-abstraction-spotcheck-v484`: replace the null low-order curriculum with dynamic
  PWA/MILP abstraction spot-checks, declarative constraint-group metadata, and false-property rejection.
- `exp5305-hardware-continuity-reachability-v484`: run KV260, PolarFire, and GateMate reachability only, with
  no speedup claim and no host `/dev/mmcblk*` assumption.
- `exp5306-capstone-v484`: synthesize positives, nulls, mixed-class harms, gated skips, hardware blocks, and
  retirement recommendations.

## Dependency Graph and Structured Gates

```text
exp5295 -> exp5296 -> exp5297

exp5297 -> exp5298 [gate: exp5297.changed_runtime_sota_ready == true]

exp5299 -> exp5300 [gate: exp5299.constraint_lns_fixture_ready == true]

exp5302 -> exp5303 [gate: exp5302.memory_policy_candidate_ready == true]

exp5301, exp5304, exp5305 are independent bounded diagnostics.

all terminal upstreams -> exp5306
```

Structured conductor gates:

- `exp5298` is gated on `exp5297.changed_runtime_sota_ready == true`.
- `exp5300` is gated on `exp5299.constraint_lns_fixture_ready == true`.
- `exp5303` is gated on `exp5302.memory_policy_candidate_ready == true`.

Prior milestone artifacts are treated as preconditions in prompts, not cross-milestone `gated_on` entries:
Exp5285 coherence fixture, Exp5287 trace fixture, Exp5289 attribution, Exp5290 fixed memory dose, Exp5291 KAN
null, Exp5292 p-bit/CDCL mixed result, and Exp5293 hardware reachability block.

## Model and Inference Requirements

Any `.484` experiment that calls an LLM must declare `MODEL_SPECS` and include at least one mandated local
SOTA GGUF model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy tiny models may be used only for CPU smoke tests and must be labeled `smoke_test_not_headline`. New
experiment scripts must use `cached_sota_pair()` or the repo's current SOTA helper where applicable and load
GGUFs through local `.gguf` paths. Do not use `AutoTokenizer.from_pretrained` on a GGUF repository.

Required inference-substrate labels for `.484`:

- `literature_ingestion_network_sources`: network-backed source refresh only.
- `live_llm_inference_changed_local_gguf_sota`: real local GGUF generation/scoring through a changed substrate
  with model id, quantization, command/backend, prompt checksum, output checksum, wall-clock receipt, and GPU
  offload receipt.
- `blocked_preconditions_with_no_quality_claim`: runtime or hardware preconditions failed; no quality claim.
- `offline_deterministic_fixture_no_llm`: fixture, parser, lexical baseline, schema, or safety-negative work
  with no live model-quality claim.
- `offline_deterministic_certificate_no_llm`: solver, MILP, KAN, factor graph, CDCL, or EBT diagnostic with no
  LLM quality claim.
- `aggregation_from_upstream_artifacts`: capstone, replay, attribution, or scheduler aggregation with no live
  model-quality claim.
- `hardware_probe_no_speedup_claim`: board reachability and environment receipts only.

## Hardware Requirements

- Local NVIDIA GPUs are required for headline SOTA GGUF inference. If unavailable, LLM-dependent tasks must
  gate-skip or emit honest blocked artifacts.
- Exp5297 must use a changed runtime substrate. Reusing the `.483` CPU-only llama-cpp-python path without new
  GPU-offload evidence is retired by `ops/exclusion_manifest.yaml`.
- KV260 is reachable via SSH only. Do not require host `/dev/mmcblk*`.
- PolarFire remains reachability/status-only unless an authenticated terminal workload already exists.
- GateMate remains physical/JTAG blocked unless the operator changes the physical setup.
- Extropic TSU/XTR-0 and Logical Kona/Aleph are reference material only. Do not claim execution,
  compatibility, or speedup without local reproducible receipts.
- Hardware tasks must record `hardware_evidence_level`, `hardware_speedup_claimed`, `blocked_reason`, and exact
  probes run.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not use `/deep-research`.
- Do not repeat Exp5284's CPU-only llama-cpp-python SOTA offload path without changed runtime substrate and new
  GPU-offload receipts.
- Do not revive the retired Phase D external generated-text/logprob scorer path.
- Do not treat grammar, JSON, schema, or trace parse validity as semantic safety or correctness.
- Do not claim hardware speedups from reachability checks, public hardware posts, or CPU-only simulations.
- Do not propose ARC solves in this milestone; no ARC level-solve task is needed for the identified gaps.
