# Research Roadmap vNEXT: Model-Local Verification and Governed Self-Evolution

**Milestone:** `2026.08.544`
**Date:** 2026-08-11
**Experiments:** Exp6310-Exp6322
**Phases:** 4
**Primary requirements:** FR7, FR11, FR12
**North-star constraint:** extract useful correctness energy from real local model state and improve a verifier-bound learner online without reopening failed shared-state, cross-family transfer, or proxy-solve paths

## Milestone thesis

V543 closed two attractive but unsupported paths. Its shared activation bus did
not survive the independent shortcut audit, and the licensed cross-family
transfer task repeated its blocked gate and is now retired. The milestone also
established two positive foundations. Reference-anchored online state learning
improved future exact outcomes without unsafe commits, and the ARC
target-licensed route passed both its causal canary and held-fold audit without
claiming a solve.

V544 keeps those boundaries. It does not repair or retry the universal bus. It
tests one correctness-energy probe per model, in each model's native state
space. A representation-surface preflight must first prove that the local GGUF
runtime exposes a causal signal that is not a norm, length, model-ID, label, or
pair-order shortcut. The expensive branch then uses exact vulnerable/fixed
Python pairs and all three mandated SOTA GGUF families. Cross-model agreement
is an outcome, never a shared representation or pooled rescue.

The self-learning branch starts from V543's positive same-domain initializer.
It turns the initializer into a versioned, factor-local policy asset. Candidate
versions face matched champion--challenger evaluation. A release can activate
only at the next task boundary and can roll back byte-for-byte to its parent.
A second experiment compares dense-feedback-directed search with repeated
sampling while protecting final validation from adaptive reuse. Base-model
weights, exact validators, and validation partitions remain immutable.

The ARC branch moves a validated route into the actual `E3AgentPolicy` only as
a default-off shadow. It measures route availability and prospective action
support on the live path. It does not inspect hidden game state, update the
solve registry, or claim a game or level solve.

## What V543 proved

| Branch | V543 evidence | V544 consequence |
|---|---|---|
| Milestone integrity | Exp6309 preserved all 13 declared states. Shared-state and licensed-transfer branches remained blocked. | Begin with a terminal handoff. Preserve branch-local promotion and exact terminal classes. |
| Source freeze | Exp6299 ended `complete_null` with zero accepted post-marker findings. | Search only after the V544 marker. A zero delta is valid, but the 2026-08-10 papers already indexed by the planner define the new scope. |
| Shared activation bus | Exp6300 declared the bus ready, but Exp6301 set `activation_bus_integrity_ready_score=0.0`. Claim-flip, disaggregated-cell, evaluator-swap, norm-only, and pair-swap controls failed. | Close the shared bus. Use independent model-local probes and repeat every failed control before any value claim. |
| Shared-state initializer | Exp6302 repeated `blocked_gate_check_failed`; Exp6303 had no deliverable. The exact retry is retired. | Do not retry a shared activation-to-ASP initializer or its live benchmark. Change both representation and task: local correctness energy over exact code pairs. |
| Continuous self-learning | Exp6304 ended `complete_positive`; the reference-anchored learner improved future exact outcomes without unsafe commits. | Extend the positive learner with version lineage, factor-local updates, delayed release, matched challengers, and exact rollback. Do not count replay as learning. |
| Licensed transfer | Exp6305 repeated `blocked_gate_check_failed` and the exact cross-family transfer retry is retired. | Keep every online update same-domain. No cross-family policy or activation transfer is scheduled. |
| Learning safety | Exp6306 ended `complete_positive` with fail-closed copied-state checks. | Preserve the safety harness and add version-parent, dense-feedback, protected-validation, and component-attribution attacks. Safety cannot promote utility. |
| ARC route | Exp6307 and Exp6308 each reported readiness `1.0`, no hidden source access, and no solve claim. Exp6307 has a live methodology warning because `random_seed` is absent. | Add the missing seed receipt, wire a default-off live-path shadow, and measure it prospectively. Do not re-solve any registry level. |

## The three largest gaps to the PRD vision

### Gap 1: no shortcut-safe model-native correctness energy

FR12 needs deterministic constraints, while the Phase 3 vision needs learned
energy over model state. The final pooled embedding and the V543 shared bus
both failed causal controls. Carnot still has no clean result showing that real
flagship model state predicts exact correctness better than prompt verdicts,
norm, length, or chance. V544 changes the scope to one probe per model and uses
held weakness families plus exact executable sidecars.

### Gap 2: continuous learning is positive but not governable as a released asset

FR11 requires improvement, retention, auditability, and rollback across an
ongoing stream. Exp6304 proved a bounded reference-anchored update, but it did
not version policies, delay activation to task boundaries, attribute changes
to constraint factors, or compare feedback-directed search with repeated
candidate sampling. V544 adds those controls without touching GGUF weights or
crossing model families.

### Gap 3: a validated ARC route is not yet present on the shipped live path

The target-license route passed canary and holdout audits, but those artifacts
do not prove that the submitted `E3AgentPolicy` can compute the route in a
fresh run. V544 adds a default-off shadow consumer with parity tests and fresh
agent-owned transition windows. This is a reachability and attribution task,
not a solve task.

## Research delta used by this roadmap

- Activation Probes (`arXiv:2608.09643`) motivates one linear security probe
  per open-weight reviewer and held weakness-type evaluation. V544 adapts the
  idea to the three mandated GGUF families and adds Carnot's failed shortcut
  controls.
- Energy-Based Constraint Networks (OpenReview `gl6l8nTXBB`) motivates global
  plus localized energy outputs. It does not establish cross-model alignment or
  replace exact validators.
- OpenLoopEvolve (`arXiv:2608.09380`) motivates version lineages,
  champion--challenger evaluation, task-boundary activation, monitoring, and
  parent rollback.
- Agentic Auto-Research is Fuzz Testing (`arXiv:2608.09855`) motivates a cheap
  dense progress signal that chooses the next update. Protected exact
  validation remains the only release authority.
- Beyond Binary (`arXiv:2608.09366`) motivates factor graphs, lazy local
  updates, and explicit movement costs for online initializer state.
- SHE (`arXiv:2608.09885`) motivates component-level failure attribution and
  safety-utility checks for every evolved asset.
- Energy-Structured Latent World Models (`arXiv:2608.09876`) motivates causal
  transition-residual receipts for the ARC shadow. The robotics results do not
  support an ARC solve claim.
- P3 (`arXiv:2608.09277`) is promoted for a later verified-code branch. V544
  records its joint program-and-proof design but does not dilute this milestone
  with a fifth branch.

The full source disposition is in `research-references.md` under the V544
planner marker.

## Target architecture

```text
               exact paired Python safety fixture
                │ vulnerable/fixed + sidecar proof
                │
      ┌─────────┴──────────┬────────────────────┐
      ▼                    ▼                    ▼
 Qwen3.6-35B MoE     Gemma-4-31B dense    Gemma-4-26B MoE
      │                    │                    │
      │ one native state   │ one native state   │ one native state
      │ surface + head     │ surface + head     │ surface + head
      └────────────── no shared bus ─────────────┘
                           │
              independent shortcut audit
                           │
          ┌────────────────┴─────────────────┐
          │ clean                            │ closed
          ▼                                  ▼
 model-local correctness energy         no value claim
          │
 prompted / final-pool / norm / length / exact-validator controls

Continuous self-learning:
sealed event ─► predecision snapshot ─► exact outcome reveal ─► factor update
                       │                         │
                  champion vN              challenger vN+1
                       └──── paired gate ─ task boundary release
                                             │
                                monitor ─ degrade ─ rollback parent

Dense guidance stream ─► progress signal ─► next candidate intervention
Protected validation  ─────────────────────► release verdict only

ARC live branch:
own attempts ─► target-license evidence ─► default-off E3 shadow proposal
      │                                           │
      └── shipped action unchanged ─ prospective support audit ─ no solve
```

## Phase 0: terminal boundary, source freeze, and representation preflight

### Exp6310: V543-to-V544 terminal transition

Consume Exp6309, the V543 operational retro, exclusion manifest, staged YAML,
and all declared artifacts. Validate 13 task IDs, dependencies, structured
gates, required prior-failure blocks, model policy, prompt endings, and
protected files. Do not activate the staged roadmap.

**Deliverable:** `results/experiment_6310_v544_terminal_transition.json`

### Exp6311: post-marker source and scope freeze

Freeze the V544 paper set, model-local probe contract, versioned same-domain
learning contract, ARC no-solve boundary, and hardware exclusions. The scan
starts after the V544 reference marker. A zero new-source delta is terminal.

**Deliverable:** `results/experiment_6311_v544_post_marker_source_scope_freeze.json`

### Exp6312: model-local representation-surface preflight

Replay the Exp5853 and Exp6301 failure ledger. Test the runtime surface on all
three mandated GGUF families before any large corpus. Prefer a true local
hidden-state surface if the local runtime exposes it with reproducible tensor
provenance. Otherwise test a preregistered output-free prefix-state trajectory.
The surface must respond to a causal code fix while defeating A/A, length,
norm, pair-order, label, truncation, and model-identity controls. It may end
cleanly null and close Phase 1.

**Deliverable:** `results/experiment_6312_model_local_representation_surface_preflight.json`

## Phase 1: model-local correctness energy

### Exp6313: exact paired code-safety fixture

Build length-matched vulnerable/fixed single-function Python pairs across
preregistered weakness families. Each pair needs compile, executable property,
AST/constraint, mutation, split, and provenance receipts. Hold out complete
weakness, repository, template, and perturbation groups. The fixture proves
only its declared properties; it is not a universal security benchmark.

**Deliverable:** `results/experiment_6313_exact_code_safety_pair_fixture.json`

### Exp6314: three-family model-local state corpus, gated on Exp6312 readiness=1

Extract the frozen representation surface for every exact pair from
Qwen3.6-35B-A3B, Gemma-4-31B, and Gemma-4-26B-A4B. Store native tensors and
paired differences separately by model. No generation, shared adapter,
cross-model normalization, or pooled rescue is allowed.

**Deliverable:** `results/experiment_6314_three_family_model_local_state_corpus.json`

### Exp6315: model-local paired-difference energy probes, gated on Exp6314 readiness=1

Fit one small linear or monotone energy head per model. Train on complete group
folds. Compare paired-difference, absolute-state, final-pooled, norm, length,
prompted-verdict, and chance controls. Report held weakness families and every
model independently.

**Deliverable:** `results/experiment_6315_model_local_paired_difference_energy_probes.json`

### Exp6316: independent model-local probe integrity audit

Reconstruct the corpus and heads from hashes. Replay claim flips, pair swaps,
label permutations, evaluator swaps, norm/length residualization, truncation,
duplicates, model identity, split leakage, and underpowered-cell controls. No
mean may hide a failed model or held family.

**Deliverable:** `results/experiment_6316_model_local_probe_integrity_audit.json`

### Exp6317: live three-family verifier benchmark, gated on Exp6316 integrity=1

Freeze the clean heads and evaluate a fresh exact holdout with all three GGUF
families. Headline value requires the local energy to beat chance and the same
model's prompted verdict on every adequately powered model fold, without
calling the exact sidecar a learned-verifier win.

**Deliverable:** `results/experiment_6317_live_three_family_model_local_verifier_benchmark.json`

## Phase 2: governed continuous self-learning

### Exp6318: versioned factor-local online initializer

Extend the positive Exp6304 initializer on a new sealed chronological stream.
Compare frozen, full-state reference-anchored, and lazy factor-local
reference-anchored arms under matched update budgets. Candidate versions have
parents, immutable snapshots, paired challenger gates, task-boundary release,
monitoring, and exact rollback. This is the milestone's required continuous
self-learning experiment. It is same-domain only.

**Deliverable:** `results/experiment_6318_versioned_factor_local_online_initializer.json`

### Exp6319: feedback-directed update search, gated on Exp6318 readiness=1

Use a cheap dense progress signal to select the next factor-update candidate.
Compare feedback-directed search with repeated sampling under identical
candidate, update, and verifier budgets. Keep final exact validation sealed
from the adaptive loop. Measure validated improvements per cost, signal
predictiveness, false discoveries, and movement cost.

**Deliverable:** `results/experiment_6319_feedback_directed_online_update_search.json`

### Exp6320: independent online self-evolution safety audit

Attack the version registry, parent links, task-boundary release, protected
validation, dense progress signal, exact outcome channel, factor attribution,
rollback, restart, poison, reversal, and forgetting controls. Audit Exp6318
even if Exp6319 is skipped or null. Safety-only success cannot promote utility.

**Deliverable:** `results/experiment_6320_online_self_evolution_safety_audit.json`

## Phase 3: ARC live shadow and capstone

### Exp6321: ARC target-licensed route live-shadow A/B

Registry-precheck every selected game and level. Add the Exp6307/Exp6308 route
as a default-off shadow on the real `E3AgentPolicy` construction path. Use only
the agent's own fresh attempts and runtime transition evidence. Compare shadow
off with shadow computed but unable to change actions. Record supported and
unsupported proposals, latency, parity, seed, and escape-hatch counts. Credit
zero solves and leave the submitted behavior unchanged.

**Deliverable:** `results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json`

### Exp6322: V544 adversarial capstone and reconciliation

Classify every declared artifact and preserve missing, flagged, null, blocked,
skipped, oracle-only, safety-only, shadow-only, and ready states. Promote each
branch independently. Reconcile OpenSpec, traceability, architecture, status,
changelog, completed research, and operational retro documents only to exact
artifact evidence.

**Deliverable:** `results/experiment_6322_v544_adversarial_capstone.json`

## Dependency graph

```text
Exp6310 terminal transition
  ├── Exp6311 source/scope freeze
  │     ├── Exp6312 representation preflight
  │     │     └─[surface=1]─► Exp6314 live state corpus
  │     │                       └─[corpus=1]─► Exp6315 local probes
  │     │                                         └► Exp6316 integrity audit
  │     │                                              └─[integrity=1]─► Exp6317 live benchmark
  │     ├── Exp6313 exact code-safety fixture ────────────────┘
  │     ├── Exp6318 versioned online initializer
  │     │     └─[readiness=1]─► Exp6319 feedback-directed search
  │     │                              └► Exp6320 safety audit
  │     └── Exp6321 ARC live-shadow A/B
  └───────────────────────────────────────────────────────────┐
                                                              ▼
                 Exp6322 capstone consumes Exp6310-Exp6321
```

The self-learning branch is independent of model-local probe promotion. The
ARC branch is independent of both. Structured gates prevent expensive live
tasks from consuming agent and GPU time after a prerequisite closes.

## Hardware and runtime requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Dual RTX 3090 CUDA host | Exp6312, Exp6314, Exp6317, Exp6321 | Use the cached mandated GGUF files. Record model hashes, tokenizer hashes, CUDA devices, layer offload, VRAM before/peak/after, actual work duration, and unload receipts. Run models sequentially when memory isolation is needed. |
| CPU and RAM | All tasks; especially Exp6313, Exp6315-Exp6320, Exp6322 | Exact Python/AST/property checks, linear heads, online updates, audits, and capstone aggregation must have deterministic CPU paths. Record peak RSS for large corpus tasks. |
| Local disk | Exp6314 and Exp6317 | Reserve at least 25 GiB before extraction. Write raw rows and checkpoints atomically. Hash every manifest and tensor shard. |
| KV260 / PolarFire / GateMate | None | No new workload or physical receipt justifies a board task. Preserve existing outcomes. |
| Extropic TSU / Z1 | None | The public tapeout announcement does not provide authenticated local access. No availability, speed, power, or sampling claim is allowed. |

The model-local heads and factor-local learner have a future sparse hardware
path: native per-model projections are matrix-vector operations; factor updates
and exact energies are sparse graph reductions. V544 measures operation counts
and memory movement on CPU/GPU. It does not claim FPGA or thermodynamic speed.

## Explicit exclusions and rerun discipline

- Do not retry the Exp6300 shared activation bus, Exp6302 shared initializer,
  or missing Exp6303 shared-state benchmark.
- Do not retry Exp6305 licensed cross-family transfer. V544 learning stays
  within one declared task domain and one version lineage.
- Do not use finite-ID generated-answer transport, parser retries, grammar
  transport, or external generated-text energy scorers.
- Do not reopen KAN replacement, MMLU-Pro final-state probing, mode-jump
  sampling, or unchanged physical-board probes.
- Do not use hidden game source, offline ground-truth BFS, per-game adapters,
  prior-game trajectories, or registry targets in the ARC task.
- Do not count exact-validator labels, exact repairs, replay hits, protected
  validation, safety-only evidence, or shadow-only ARC proposals as product
  gains.
- Every scope-adjacent task carries the prior honest verdict, a concrete
  mechanism change, and `retire_if_same_verdict: true` in the staged YAML.

## Milestone success criteria

V544 succeeds as a research milestone if it produces decision-grade terminal
evidence, including a clean null. Branch promotion is stricter:

1. **Model-local verification:** every mandated model passes the independent
   shortcut audit, and its frozen local energy beats chance and its own
   prompted-verdict baseline on adequately powered fresh exact holdouts.
2. **Continuous self-learning:** a versioned factor-local learner improves
   future-event utility over frozen state, does not regress the full-state
   anchored control, reduces update movement, releases only at boundaries, and
   rolls back exactly with zero unsafe commits.
3. **Feedback-directed search:** protected validation confirms more genuine
   improvements per matched cost than repeated sampling, and the dense signal
   never acts as release authority.
4. **ARC reachability:** the actual `E3AgentPolicy` computes the target-licensed
   route in a default-off shadow on fresh agent-owned windows, preserves action
   parity, accesses no escape hatch, and credits zero solves.
5. **Evidence integrity:** every declared artifact is terminal or explicitly
   missing; all required commands, durations, substrates, seeds, hashes,
   principles, provenance, and adversarial states remain visible in Exp6322.

No aggregate score may rescue a failed model, fold, safety gate, or provenance
cell. A clean refutation closes its branch and informs the next milestone.
