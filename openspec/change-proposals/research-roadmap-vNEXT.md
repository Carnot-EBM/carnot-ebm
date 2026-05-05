# Research Roadmap vNEXT: Milestone 2026.04.103

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.102 SOTA Runtime Recovery + Certifiable Constraint Learning + Hardware-Portability Audits
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .102 Proved

Milestone 2026.04.102 reached terminal conductor state for every scheduled
task. Its retrospective scored 11 of 14 criteria as met. The milestone did not
finish the certificate branch, but it did establish the first credible local
SOTA runtime baseline and clarified the next blockers.

- `exp1309` repaired the SOTA GGUF resolver: two cached mandated headline
  models now produce loadable specs. The optional middle MoE,
  `unsloth/gemma-4-26B-A4B-it-GGUF`, remained absent.
- `exp1310` smoke-loaded the resolved SOTA pair through the local llama.cpp
  path. Two models loaded and produced tokens, with aggregate throughput around
  20.2 tokens/second on the tiny smoke prompts.
- `exp1311` showed answer stability can look good while the runtime still
  returns unusably short generations. `answer_stability_score=0.9`, but
  many raw outputs were empty or one token and `pysat_verified_rate=0.525`.
- `exp1312` compared triggered certificates, DCCD, GBNF, and repair. DCCD beat
  grammar-only by `0.2`, repair success was `1.0`, and grammar/projection tax
  favored compact prompts. However `certificate_parse_rate=0.71223`, below the
  `0.75` gate, and `certificate_truthfulness_rate=0.69697`.
- `exp1313`, `exp1314`, and `exp1316` were correctly gate-blocked by the parse
  shortfall, so semantic validator, safe-prefix acceptance, and DVI
  certificate-tail updates remain unproven.
- `exp1315` advanced continuous self-learning: non-forgetting replay passed
  with `nonforgetting_certificate_rate=1.0`, `memory_regression_count=0`, and
  `self_learning_delta_overall=1.596429`, but it stayed non-headline because it
  was not connected to fresh SOTA certificate tails.
- `exp1317` produced a positive GRPO/VPRM replay headline gate
  (`grpo_vprm_delta=0.45`), but it was still replay-only.
- `exp1318` learned a stop-policy artifact on held-out replay. Precision and
  recall were perfect on that split, but the delta over the conservative replay
  policy was `0.0`.
- `exp1319` and `exp1320` produced honest hardware-portability artifacts for
  KAN and p-bit sampling. They did not make hardware-execution claims.
- `exp1321` kept publication in operator hold and added related-work deltas.
  `exp1322` named the main carry-forward: recover certificate parse quality,
  then rerun the semantic, safe-prefix, and DVI branches.

The natural next milestone is therefore not another broad expansion. It is a
parse-and-semantics recovery milestone: diagnose short SOTA generations, raise
certificate parse quality above the existing gate, bind parsed certificates to
semantic validators, and then connect continuous self-learning to certificate
tails.

## Research Signals Added Before Planning

The 2026-05-05 post-.102 planning sweep added recent 2025-2026 references to
`research-references.md` before this roadmap was designed. The items most
directly shaping `.103` are:

- Reality Check of LLMs as Formalizers on CSPs (`arXiv:2505.13252`): LLM
  formalization can fail on real CSPs even when surface answers look stable.
  `.103` therefore separates answer agreement, certificate parseability,
  semantic validity, and solver-backed truth.
- SatIR (`arXiv:2604.08849`): scalable high-recall constraint indexing suggests
  a practical path for routing parsed certificates into semantic validator
  components instead of monolithic string checks.
- Orthographic Constraint Satisfaction (`arXiv:2511.21086`): hard symbolic
  constraints expose model-family and generation-control failures that ordinary
  QA benchmarks can hide.
- H-Neurons (`arXiv:2512.01797`) and follow-on cross-domain transfer work
  (`arXiv:2604.19765`): hallucination probes must be domain calibrated. `.103`
  avoids universal-detector claims.
- Real-time hallucinated-entity detection (`arXiv:2509.03531`) and token-level
  entropy production rate (`arXiv:2509.04492`): token-health, entropy, and
  top-k/logprob diagnostics are now first-class runtime checks.
- p-DNN sampling (`arXiv:2507.07763`): sampling-versus-bits accounting is a
  better near-term hardware bridge than pretending local FPGA or TSU execution
  is already available.
- Current code artifacts worth tracking: EBT, Cactus, constrained diffusion,
  and LUT-KAN provide implementation examples for future experiments, but are
  not treated as validated Carnot dependencies.

## Three Biggest Gaps

1. **SOTA generation and certificate completeness gap.** `.102` proved the
   local SOTA pair can load, but certificate tasks still saw empty or one-token
   outputs and parse rate below gate. Carnot needs a token-health diagnostic and
   runtime/prompt repair before any semantic validator result can be trusted.

2. **Semantic acceptance gap.** The PRD vision requires verifiable reasoning,
   not parseable JSON alone. The semantic validator, MUS repair, safe-prefix,
   and low-risk acceptance branches have not yet run on fresh parseable SOTA
   certificates.

3. **Continuous self-learning evidence gap.** Non-forgetting replay is strong,
   but FR-11 needs verifier-governed self-learning connected to new certificate
   tails. `.103` must bridge CerCE-style memory control into DVI-style online
   updates without accepting new violations.

Hardware remains a fourth, explicit non-goal for headline claims. `.103` should
improve hardware energy accounting and portability packets while continuing to
forbid FPGA, KV260, TSU, ROCm, or Kona execution claims unless real local
execution occurs during the experiment.

## Architecture Target

```text
Phase 0: runtime and certificate diagnostics
  exp1323 SOTA token-health and prompt/runtime diagnostic
      |
      +--> exp1324 certificate failure taxonomy / CSP formalizer audit
      |
      v
Phase 1: certificate parse recovery and semantic acceptance
  exp1325 runtime-fixed DCCD/GBNF certificate extraction v5
      |
      v
  exp1326 SatIR/NSVIF semantic validator + MUS repair
      |
      v
  exp1327 BEAVER-lite/Cactus safe-prefix acceptance

Phase 2: verifier-governed continuous self-learning
  exp1328 memory promotion and non-forgetting v2
      |
      +--> exp1329 DVI certificate-tail online update
              |
              v
          exp1330 GRPO/VPRM v12 micro-audit

Phase 3: constraint coverage, calibrated probes, hardware accounting, closeout
  exp1331 orthographic hard-constraint smoke benchmark
  exp1332 domain-calibrated token/EPR probe
  exp1333 p-DNN samples-vs-bits energy accounting
  exp1334 LUT-KAN reproducibility baseline
  exp1335 publication hold + related-work delta v12
  exp1336 milestone retro and carry-forward plan
```

## Phase 0: Runtime and Certificate Diagnostics

Goal: identify whether `.102` certificate failure was mainly generation
configuration, prompt shape, parser coverage, semantic invalidity, or model
behavior.

- `exp1323-sota-gguf-token-health-prompt-runtime-diagnostic`: run tiny local
  SOTA probes with the mandated GGUF pair and record empty/one-token rate,
  generation settings, entropy/logprob availability, and prompt variants.
- `exp1324-certificate-failure-taxonomy-formalizer-reality-check`: audit `.102`
  certificates against solver-backed labels and the new CSP formalizer reality
  literature. Produce a concrete taxonomy of parse failures, semantic failures,
  undergeneration, and hardcoded-solution leakage.

Success bar: `exp1323` must either recover multi-token generation or emit a
terminal blocker that prevents downstream SOTA calls from wasting time.
`exp1324` must name which failure modes can plausibly be repaired in `.103`.

## Phase 1: Certificate Parse Recovery and Semantic Acceptance

Goal: rerun only the blocked certificate branch with explicit prior-failure
handling and structured gates.

- `exp1325-triggered-certificate-extraction-v5-runtime-fixed-dccd-gbnf`: gated
  on token recovery. Rerun DCCD/GBNF extraction with the fixed runtime/prompt
  settings and require `certificate_parse_rate >= 0.75` before semantic work.
- `exp1326-satir-nsvif-semantic-validator-gated-on-parse-ge-075`: gated on
  `exp1325`. Compile parsed certificates into indexed semantic validator
  components, preserve unknown states, and add MUS repair hints.
- `exp1327-beaver-lite-cactus-safe-prefix-gated-on-validator-pass`: gated on
  validator execution. Measure false acceptance, verifier-call reduction, and
  safe-prefix repair delta.

Success bar: the branch either reaches a semantically checked certificate path
with bounded false acceptance, or it produces a narrower blocker than `.102`:
token health, parseability, semantic validity, or acceptance risk.

## Phase 2: Verifier-Governed Continuous Self-Learning

Goal: convert replay-only self-learning evidence into verifier-controlled
certificate-tail updates while preserving non-forgetting.

- `exp1328-continuous-self-learning-memory-promotion-v2`: mandatory
  self-learning task. Rerun memory promotion/demotion with CerCE-style
  non-forgetting checks and headline certificate replay bookkeeping.
- `exp1329-dvi-certificate-tail-online-update-v2-gated-on-parse-and-nonforgetting`:
  gated on parse recovery and non-forgetting. Apply DVI-style updates to
  certificate tails and report acceptance deltas without claiming lossless
  improvement unless the evidence warrants it.
- `exp1330-grpo-vprm-v12-micro-audit-gated-on-dvi-lossless`: gated on DVI and
  positive self-learning evidence. Keep this as a micro-audit unless the gates
  justify fresh policy work.

Success bar: at least one self-learning artifact reports non-forgetting,
positive verifier-controlled delta, and no increase in accepted violations. If
the parse gate blocks DVI again, `.103` must retire or narrow the DVI path
rather than blindly rerun it.

## Phase 3: Coverage, Hardware Accounting, Publication State, Retro

Goal: add coverage for hard constraints and hallucination probes, then close the
milestone with honest hardware and publication state.

- `exp1331-orthographic-hard-constraint-smoke-gated-on-token-recovery`: use the
  orthographic constraint literature to test whether SOTA GGUF generation
  control obeys exact hard constraints.
- `exp1332-domain-calibrated-token-epr-probe-gated-on-topk`: use token-health,
  token-level EPR, and hallucination-probe findings to build a calibrated probe
  audit. No universal hallucination-detector claim is allowed.
- `exp1333-pdnn-samples-vs-bits-energy-accounting`: build a CPU accounting
  artifact that compares sampling effort to weight-bit precision for a tiny
  verifier.
- `exp1334-lut-kan-reproducibility-baseline`: reproduce a small LUT-KAN-style
  accounting baseline connected to `.102` KAN hardware-complexity audit.
- `exp1335-publication-hold-related-work-delta-v12`: keep publication state
  honest and fold in `.103` literature deltas.
- `exp1336-milestone-103-retro-carryforward`: score `.103`, reconcile docs,
  and name the next carry-forward tasks.

Success bar: constraint coverage broadens without overclaiming, hardware work
stays at accounting/portability scope unless real hardware runs, publication
remains in the correct operator state, and the retro has enough detail to plan
the next milestone.

## Dependency Graph

```text
exp1323 ---> exp1325 ---> exp1326 ---> exp1327
   |            |             |
   v            |             +----------------+
exp1324         |                              |
                v                              v
exp1328 ----> exp1329 --------------------> exp1330

exp1323 -------------------------------> exp1331
exp1323 -------------------------------> exp1332

exp1320 context -----------------------> exp1333
exp1319 context -----------------------> exp1334
exp1321 + new refs --------------------> exp1335

exp1323..exp1335 ----------------------> exp1336
```

## Hardware Requirements

- SOTA LLM tasks must use local mandated GGUF models through
  `cached_sota_pair(gpu_indices=(0, 1))` or an equivalent
  `scripts/experiment_template.py` helper. Every fresh LLM experiment must
  include at least one of:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`,
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- Legacy small models such as Qwen3.5-0.8B or gemma-4-E4B-it may only be CPU
  smoke tests and must set `headline_result_allowed=false`.
- Phase 0, Phase 1, and hard-constraint coverage tasks expect the local
  dual-GPU workstation and llama.cpp-compatible runtime already proven in
  `.102`. They must record model IDs, quantization, generation settings,
  token counts, and any GPU memory/throughput data available.
- Phase 3 hardware tasks are CPU accounting or portability artifacts. They may
  read KAN and p-bit design packets, but may not claim FPGA, KV260, ROCm, TSU,
  or Kona execution unless the corresponding hardware and runtime are actually
  exercised during the task.

## Success Criteria

- `exp1323` recovers multi-token SOTA generations or cleanly blocks downstream
  SOTA work with a precise runtime cause.
- `exp1325` reaches `certificate_parse_rate >= 0.75` or retires the current
  DCCD/GBNF recovery path with evidence.
- `exp1326` executes semantic validators on parsed certificates and preserves
  unknown states.
- `exp1327` reports false-acceptance risk before claiming safe-prefix savings.
- `exp1328` preserves non-forgetting and keeps accepted violations from rising.
- `exp1329` runs only if parse and non-forgetting gates pass.
- `exp1330` runs only if DVI produces evidence strong enough to justify the
  GRPO/VPRM micro-audit.
- `exp1331` and `exp1332` broaden hard-constraint and calibrated-probe evidence
  without universal claims.
- `exp1333` and `exp1334` improve hardware accounting while keeping
  `hardware_claim_allowed=false` unless real hardware executes.
- `exp1335` keeps publication in the correct operator state.
- `exp1336` reconciles `ops/status.md`, `ops/changelog.md`, and
  `_bmad/traceability.md` with the evidence actually produced.

## Decentralization Implication

The `.103` plan keeps Carnot aligned with the long-term decentralization goal:
verification should move toward compact, locally executable certificates and
energy-accounted samplers, not opaque remote model calls or unverifiable
central services. The milestone is still an evidence-building step, not a
deployment claim.
