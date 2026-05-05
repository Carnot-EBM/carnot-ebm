# Research Roadmap vNEXT: Milestone 2026.04.102

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.101 Activation Hygiene + SOTA Certificate v3 + Online Self-Learning Control
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .101 Proved

Milestone 2026.04.101 completed all scheduled tasks, but its retrospective
artifact scored 8 of 13 criteria as met. The missing criteria were concentrated
in the SOTA certificate branch, which remained blocked by runtime/cache
readiness rather than by negative certificate evidence.

- `exp1296` proved the prior-failure and structured-gate activation audits can
  pass before a milestone activates. This fixed the `.100` DOOMED_RERUN_BLOCK
  waste pattern.
- `exp1297` showed the SOTA path is close but still not usable:
  `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` were visible
  in cache metadata, but `cached_sota_pair()` returned no loadable specs and the
  missing `unsloth/gemma-4-26B-A4B-it-GGUF` kept `cached_sota_ready=false`.
- `exp1298`, `exp1299`, `exp1300`, `exp1301`, and `exp1304` therefore did not
  produce headline certificate, routing, Cactus, or GRPO evidence. They are
  carry-forwards with explicit gate repair, not simple reruns.
- `exp1302` and `exp1303` produced the strongest new science signal:
  sandboxed skill-graph candidates plus positive online memory-policy replay
  (`self_learning_delta_overall=1.596429`) with fewer accepted violations.
  The result is still non-headline because it did not use fresh SOTA
  certificate data and did not prove non-forgetting.
- `exp1305` confirmed HardNet++ remains better than SnareNet on replay
  (`hardnetpp_delta_over_snarenet=1.2207`) and that DSP feasibility is a useful
  operator gate (`feasibility_channel_auc=0.6605`), but it did not learn a
  general stop policy.
- `exp1306` completed a local energy-bridge alignment artifact, but not a
  strategic implementation path. `exp1307` kept publication in operator hold
  with no credentialed arXiv submission.

The natural next milestone is therefore a recovery-and-certification milestone:
repair the local SOTA resolver, run the smallest defensible headline certificate
path, bind parsed certificates to semantic validators, and convert the `.101`
self-learning signal into a non-forgetting controlled loop.

## Research Signals Added Before Planning

The 2026-05-05 planning sweep added recent 2025-2026 references to
`research-references.md` before this roadmap was designed. The items most
directly shaping `.102` are:

- ConstraintBench (`arXiv:2602.22465`), SATQuest (`arXiv:2509.00930`), and
  Compact Constraint Encoding (`arXiv:2604.07192`) give small but grounded
  benchmark slices for certificate truthfulness and constraint translation.
- ConstrainPrompt (ICLR 2026 submission) and NSVIF (`arXiv:2601.17789`) suggest
  schema-first validators, verifier-in-the-loop correction, and uncertainty
  handling instead of syntax-only certificate acceptance.
- Residual Drift and MUS-Repair (ICLR 2026 submission) provide a concrete
  repair channel for unsatisfied constraints and residual error localization.
- CerCE (ICLR 2026 submission) motivates explicit non-forgetting certificates
  for continuous self-learning.
- Dynamic Verifier Integration (DVI, `arXiv:2510.05421`) provides the right
  online-update target once parseable certificate tails exist.
- p-bit dual-BRAM annealer work, p-bit update-dynamics results, KAN hardware
  complexity work, lmKAN, and physical analog KANs inform hardware-portability
  audits without pretending that local FPGA or TSU bring-up is unblocked.

## Three Biggest Gaps

1. **Local SOTA runtime gap.** The PRD requires credible local SOTA verifier
   experiments. `.101` found two mandated headline models in cache metadata, but
   the helper path still returned no usable model specs. Until the resolver can
   produce a two-model SOTA pair and a smoke load can run, every downstream
   headline certificate task will continue to gate out.

2. **Certificate semantics gap.** Carnot still lacks a measured chain from LLM
   output to parseable certificate to semantic validator to safe acceptance.
   The prior milestones selected grammar tooling and sketched routing, but did
   not prove certificate truthfulness, MUS repair, verifier-backed unknown
   states, or false-acceptance bounds on fresh SOTA outputs.

3. **Controlled self-learning gap.** `.101` showed online memory policy can
   improve replay metrics, but FR-11 needs continuous self-learning that is
   verifier controlled, non-forgetting, and connected to certificate tails.
   The next step is not more promotion-only memory. It is CerCE-style
   non-forgetting checks plus DVI-style verifier integration.

A fourth gap is hardware portability. The current repair and KAN/p-bit evidence
is promising but local hardware claims must remain design-packet/audit claims
until Vivado/KV260/TSU or equivalent real hardware is available.

## Architecture Target

```text
Phase 0: SOTA runtime recovery
  exp1309 resolver repair
      |
      v
  exp1310 llama.cpp smoke load and throughput probe
      |
      v
Phase 1: certifiable SOTA constraint reasoning
  exp1311 ConstraintBench/SATQuest answer stability
      |
      v
  exp1312 DCCD + GBNF triggered certificate extraction
      |
      +-----------------------------+
      |                             |
      v                             v
  exp1313 semantic validators       exp1314 BEAVER-lite/Cactus acceptance
  ConstrainPrompt/NSVIF/MUS         safe-prefix and low-risk filters

Phase 2: continuous self-learning
  exp1302/exp1303 .101 memory evidence
      |
      v
  exp1315 CerCE non-forgetting audit
      |
      +-----------------------------+
      |                             |
      v                             v
  exp1316 DVI certificate-tail       exp1317 GRPO/VPRM v11
  online update                     only if cert + learning gates open

Phase 3: repair, hardware portability, publication state
  exp1318 learned repair stop policy
  exp1319 KAN hardware complexity audit
  exp1320 p-bit sampler portability packet
  exp1321 publication hold + related-work delta
  exp1322 milestone retro
```

## Phase 0: SOTA Runtime Recovery

Goal: turn the `.101` cache finding into a usable local SOTA pair. The missing
middle MoE model should be recorded, but two cached mandated headline models
should be enough to run a two-model certificate study if both load.

- `exp1309-sota-gguf-pair-resolver-repair`: inspect and, if necessary, patch
  `cached_sota_pair()` so two cached mandated GGUF models produce two loadable
  specs. If this changes intended behavior, update the relevant OpenSpec
  requirement before implementation. Add focused tests around the resolver.
- `exp1310-sota-gguf-llamacpp-smoke-load`: gated on `exp1309`, run the smallest
  llama.cpp smoke load and throughput probe for the resolved pair. This is a
  headline gate, not a benchmark race.

Success bar: `sota_pair_ready=true`, two mandated model specs are returned, and
`headline_result_possible=true` after smoke loading. If the cache is still
unusable, later tasks skip before spending model time.

## Phase 1: Certifiable SOTA Constraint Reasoning

Goal: measure the complete certificate path on fresh local SOTA outputs and only
then accept or reject constrained claims.

- `exp1311-sota-constraintbench-satquest-answer-stability`: use a tiny
  verifier-backed micro-slice from ConstraintBench/SATQuest-style tasks to
  measure cross-model stability, disagreement, PySAT/Z3 verification rate, and
  feasibility/unknown handling.
- `exp1312-triggered-certificate-extraction-dccd-gbnf`: compare raw triggered
  certificates, grammar-constrained GBNF certificates, DCCD-style compact
  encoding, and repair. Measure parse rate, truthfulness, repair success, and
  grammar/projection cost.
- `exp1313-constrainprompt-nsvif-semantic-validator-mus-repair`: compile parsed
  certificates into executable validators, track semantic violations, use
  MUS-style repair hints for failures, and record residual-drift cases.
- `exp1314-beaver-lite-cactus-safe-prefix-acceptance`: run low-risk acceptance
  only if parse and validator gates pass. Measure false acceptance, verifier
  call reduction, safe-prefix repair delta, and risk-bound proxy.

Success bar: a parsed, truthful, semantically checked certificate path either
reaches headline eligibility or emits a precise blocker with false-acceptance
and unknown-state evidence.

## Phase 2: Continuous Self-Learning

Goal: promote the `.101` self-learning signal into a verifier-controlled loop
with explicit non-forgetting and certificate-tail online updates.

- `exp1315-continuous-self-learning-cerce-nonforgetting-audit`: run the
  mandatory continuous self-learning experiment. Use `.101` memory evidence,
  replay cases, and CerCE-style checks to measure non-forgetting, memory
  regressions, accepted-violation deltas, and Lagrangian penalties.
- `exp1316-dvi-certificate-tail-online-update`: gated on parseable certificates
  and non-forgetting, apply DVI-style updates to certificate tails and report
  acceptance deltas without claiming lossless improvement unless the evidence
  warrants it.
- `exp1317-grpo-vprm-v11-headline-gate`: run only if certificate and
  self-learning gates open. This keeps expensive policy optimization from
  repeating `.101` gate-blocked work.

Success bar: at least one self-learning artifact reports non-forgetting evidence
and a positive verifier-controlled delta. If SOTA certificates remain blocked,
the self-learning branch can still produce replay-only non-headline evidence.

## Phase 3: Repair, Hardware Portability, Publication State, Retro

Goal: convert promising but local signals into honest engineering artifacts.

- `exp1318-hardnetpp-dsp-learned-stop-policy`: move from a replay operator gate
  to a learned stop/continue policy with a held-out split and explicit precision
  and recall.
- `exp1319-kan-hardware-complexity-audit`: audit KAN verifier/repair candidates
  using recent KAN hardware-complexity findings and local RM/BOP/NABS-style
  accounting. This is a portability audit, not a hardware result.
- `exp1320-pbit-sampler-portability-packet`: write a p-bit dual-BRAM/reuse
  factor design packet and CPU equivalence check for future FPGA work, avoiding
  blocked Vivado/KV260 claims.
- `exp1321-publication-hold-related-work-delta-v11`: update publication state
  and related-work delta without credentialed arXiv submission while the
  operator hold remains active.
- `exp1322-milestone-retro-102`: score `.102`, identify carry-forwards, and
  reconcile status/changelog/traceability notes.

Success bar: repair generalization is measured, hardware remains honestly
scoped, publication state is terminal, and the retro names only evidence-backed
carry-forwards.

## Dependency Graph

```text
exp1309 ---> exp1310 ---> exp1311 ---> exp1312 ---> exp1313 ---> exp1314
                                      |              |
                                      |              +--> exp1318
                                      |
exp1302/exp1303 context ---> exp1315 +--> exp1316 ---> exp1317

exp1305 context ------------------------------> exp1318
KAN/p-bit prior hardware context -------------> exp1319
KV260/Vivado blocked context -----------------> exp1320
exp1307 publication hold ---------------------> exp1321

exp1309..exp1321 -----------------------------> exp1322
```

## Hardware Requirements

- Phase 0 and Phase 1 SOTA tasks require the local mandated GGUF cache plus a
  llama.cpp-compatible runtime. The planned headline pair is whichever two
  loadable specs `cached_sota_pair(gpu_indices=(0, 1))` returns from:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- GPU tasks assume the existing dual-GPU local workstation profile. Smoke tests
  should record GPU memory and throughput, but not turn this milestone into a
  performance benchmark.
- CPU-only fallback is allowed only for smoke tests or replay artifacts and must
  set `headline_result_allowed=false` when legacy small models are used.
- No experiment may claim FPGA, KV260, ROCm, Extropic TSU, or Kona hardware
  execution unless that hardware/runtime is actually available during the run.
  Hardware work in `.102` is limited to portability packets and complexity
  audits.
