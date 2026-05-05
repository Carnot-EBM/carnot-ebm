# Research Roadmap vNEXT: Milestone 2026.04.104

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.103 SOTA Certificate Parse Recovery + Semantic Acceptance + Verifier-Governed Self-Learning
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .103 Proved

Milestone 2026.04.103 reached operator closeout, but the available local
artifacts show that the scientific branch only partially advanced. This is a
useful result: it separates the next blocker into an environment/scheduler
failure class and a certificate-method class.

| Track | Evidence | Finding |
|---|---|---|
| SOTA token health | `exp1323` | The local mandated SOTA GGUF pair can produce multi-token outputs when prompt/runtime settings are corrected. `min_tokens_recovered=true`; Qwen3.6-35B-A3B and Gemma4-31B both ran through llama.cpp; top-k/logprob and entropy proxy data were available. |
| Certificate failure taxonomy | `exp1324` | The `.102` parse gate missed by `0.03777`; at least 6 additional parseable attempts were needed. Dominant classes were parser/schema mismatch, undergeneration, semantic invalidity, solver disagreement, UNKNOWN mishandling, and possible hardcoded-solution leakage. |
| Certificate rerun | `exp1325` | The intended runtime-fixed DCCD/GBNF rerun did not reach substantive execution; local artifact stayed `status=in_progress` after conductor disk-quota failures. This is an environmental/scheduler failure, not negative evidence about trigger-switched certificates. |
| Semantic/DVI branches | `exp1326`, `exp1327`, `exp1329`, `exp1330` | Downstream tasks gate-blocked or were skipped because `exp1325` never produced the parse gate evidence. These are not valid scientific failures and must not be blindly rerun without a new prerequisite structure. |
| Operations | `results/operational_retro_2026_04_103.json` | The next loop needs disk-quota/inode preflight, repeated pre-test signature quarantine, terminal blocked artifacts, and dependency pruning before expensive SOTA tasks launch. |

The natural next milestone is therefore a recovery-and-replacement milestone:
make the environment state explicit, close stale `.103` skeletons as scheduler
facts, and rerun certificate extraction with a materially different method:
trigger-before-constrain plus dynamic certificate grammars. Only after that
should semantic constrained validation, verifier-cost scheduling, DVI, and GRPO
run.

## Research Signals Added Before Planning

The post-.103 sweep added the following 2025-2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2601.07525`, "Thinking Before Constraining": trigger-switched
  generation preserves reasoning before entering structured output mode.
- `arXiv:2601.04426`, "XGrammar 2": dynamic grammar dispatch, JIT, and grammar
  caching for branchy structured generation.
- `arXiv:2509.00360` / POPL 2026, "ChopChop": semantic constrained decoding
  over program structures rather than syntax-only token masks.
- `arXiv:2601.18753`, "HalluGuard": separates data-driven hallucinations from
  reasoning-driven instability, which maps cleanly onto Carnot's certificate
  failure taxonomy.
- `arXiv:2601.15498` and `arXiv:2601.23180`, MARS and TriSpec: verification
  cost should be margin-aware and proxy-assisted, not a binary "always call
  full verifier" decision.
- `arXiv:2505.23061` and `arXiv:2508.10111`: diffusion-LLM constrained
  inference provides a fallback if autoregressive certificate tails remain
  brittle.
- Semantic Scholar's EBT citation neighborhood: EBT, NRGPT, EBT-Policy,
  metacognitive EBT code generation, and intrinsic-optimizer Transformer work
  raise the bar for Carnot's Phase-3 claims.
- Extropic's current public XTR-0/Z1/THRML status and Logical Intelligence's
  Kona 1.0 positioning: Carnot's differentiator must be local, open,
  reproducible certificates and hardware-portable energy evaluation.

## Three Biggest Gaps

1. **Environmental reliability gap.** `.103` did not fail because the models
   could not run; it failed because disk quota and repeated pre-test signatures
   turned into scheduler churn. The next milestone must first make these
   states terminal and measurable without modifying `scripts/research_conductor.py`.

2. **Certificate semantics gap.** `.102` and `.103` show parseable JSON is not
   enough. Carnot needs trigger-switched certificate tails, dynamic grammars
   that preserve UNKNOWN/proof/repair branches, and semantic validators that
   execute or prove properties.

3. **Verifier-governed self-learning gap.** Non-forgetting replay is strong,
   but FR-11 still lacks fresh certificate-tail updates. `.104` must connect
   self-learning to verifier failure types while keeping accepted violations
   from increasing.

## Architecture Target

```text
Phase 0: environment and .103 state closure
  exp1337 environment gate audit
      |
      +--> exp1338 .103 skeleton/gate-state finalizer
      |
      v
Phase 1: trigger-switched semantic certificates
  exp1339 dynamic grammar / TagDispatch dry-run
      |
      v
  exp1340 trigger-before-constrain certificate extraction v6
      |
      +--> exp1341 HalluGuard-style failure split
      |
      v
  exp1342 semantic constrained validator pilot
      |
      v
  exp1343 margin-aware BEAVER/Cactus verifier scheduler

Phase 2: verifier-governed self-learning
  exp1344 failure-type memory policy and non-forgetting
      |
      +--> exp1345 DVI certificate-tail update v3
              |
              v
          exp1346 GRPO/VPRM v13 micro-audit

Phase 3: hardware/architecture accounting and closeout
  exp1347 THRML compatibility audit
  exp1348 p-bit update-dynamics / dual-BRAM packet v2
  exp1349 EBT citation + Kona parity gap audit
  exp1350 milestone retro and carry-forward plan
```

## Phase 0: Environment and .103 State Closure

Goal: stop spending SOTA or semantic-validator slots while the environment is
not capable of writing terminal artifacts or passing focused preflight.

- `exp1337-environment-gate-disk-pretest-stale-skeleton-audit`: produce a
  single artifact describing disk quota, inode state, free-space thresholds,
  repeated pre-test signature, and stale `.103` skeleton artifacts. This is an
  audit/preflight task only; it must not modify `scripts/research_conductor.py`.
- `exp1338-exp1325-skeleton-and-gate-state-finalizer`: turn the `.103`
  in-progress/gate-block evidence into a clean carry-forward plan for `.104`,
  including the minimum parse recovery needed and which downstream gates should
  stay closed until new evidence exists.

Success bar: `environment_ready=true` and `certificate_recovery_ready=true`, or
the milestone terminates the expensive certificate branch with a precise
environment blocker.

## Phase 1: Trigger-Switched Semantic Certificates

Goal: recover the certificate branch using techniques that are materially
different from the failed `.103` DCCD/GBNF rerun.

- `exp1339-xgrammar2-tagdispatch-certificate-grammar-dryrun`: build a local
  dynamic grammar plan for UNKNOWN/SAT/UNSAT/proof/repair certificate sections
  and measure grammar state transitions without fresh SOTA generation.
- `exp1340-trigger-before-constrain-certificate-v6-sota`: run the SOTA GGUF
  certificate extraction with free-form reasoning until a trigger token, then
  structured certificate-tail generation. Compare against `.102` DCCD/GBNF.
- `exp1341-halluguard-certificate-failure-split`: separate data-driven,
  reasoning-driven, undergeneration, parser, and semantic failures using
  `exp1323`, `exp1324`, and any `exp1340` cases.
- `exp1342-chopchop-nsvif-semantic-validator-gated`: when parse rate clears
  the gate, compile parsed certificates into executable semantic validators
  inspired by NSVIF/SatIR/ChopChop.
- `exp1343-margin-aware-beaver-cactus-scheduler`: when semantic validation
  works, evaluate verifier-call reduction with low-margin escalation and false
  acceptance accounting.

Success bar: either `certificate_parse_rate >= 0.75` with semantic validator
execution, or a narrowed blocker that is not "rerun DCCD again."

## Phase 2: Verifier-Governed Continuous Self-Learning

Goal: satisfy the research program's continuous self-learning requirement while
guarding against memory corruption and accepted-violation growth.

- `exp1344-continuous-self-learning-failure-type-memory-policy`: mandatory
  self-learning task. Use failure-type labels from `.103/.104` to decide memory
  promotion/demotion, preserve non-forgetting, and avoid increasing accepted
  violations.
- `exp1345-dvi-certificate-tail-v3-gated`: apply DVI-style online updates only
  after parse, semantic validation, and non-forgetting gates pass.
- `exp1346-grpo-vprm-v13-gated-micro-audit`: run a bounded GRPO/VPRM replay or
  micro-audit only if DVI produces lossless-acceptance evidence.

Success bar: the self-learning artifact runs unconditionally and reports
non-forgetting plus accepted-violation deltas. DVI/GRPO remain gated.

## Phase 3: Hardware/Architecture Accounting and Closeout

Goal: keep Phase-2/Phase-3 strategy current without overclaiming.

- `exp1347-thrml-compatibility-parity-audit`: test whether Carnot's tiny
  Ising/KAN verifier cases can be represented in Extropic's THRML simulation
  interface, or document the exact missing dependency.
- `exp1348-pbit-update-dynamics-dual-bram-packet-v2`: update the p-bit
  portability packet with sync/async update dynamics, reuse factor, BRAM layout,
  and DAC precision assumptions.
- `exp1349-ebt-citation-kona-parity-gap-audit`: map the EBT citation
  neighborhood and Kona 1.0 public positioning to Carnot's local evidence and
  Phase-3 obligations.
- `exp1350-milestone-104-retro-carryforward`: evaluate the milestone, keep
  publication hold honest, and name carry-forwards with prior-failure hygiene.

Success bar: hardware work stays at simulation/accounting scope unless local
hardware actually executes, and the retro creates a clean `.105` handoff.

## Dependency Graph

```text
exp1337 ---> exp1339 ---> exp1340 ---> exp1342 ---> exp1343
    |           |            |            |
    v           |            v            |
exp1338 --------+        exp1341          |
                             |            |
exp1344 ---------------------+----------> exp1345 ---> exp1346

exp1320 + new p-bit refs ----------------> exp1348
Extropic THRML status -------------------> exp1347
EBT/Kona references ---------------------> exp1349
exp1337..exp1349 ------------------------> exp1350
```

## Hardware Requirements

- Fresh LLM-bearing tasks must use local mandated GGUF models through
  `cached_sota_pair(gpu_indices=(0, 1))` or an equivalent helper. Each fresh
  LLM task must include at least one of:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`,
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- Legacy small models such as Qwen3.5-0.8B and gemma-4-E4B-it may only be
  CPU smoke tests with `headline_result_allowed=false`.
- `exp1340` is the only planned fresh SOTA generation task. It should use both
  RTX 3090s when possible and record model IDs, quantization, GPU assignment,
  token counts, logprob availability, and throughput.
- Hardware tasks are THRML simulation or portability/accounting tasks. They may
  not claim FPGA, KV260, TSU, ROCm, analog, or Kona execution unless that
  runtime is actually exercised.

## Success Criteria

- `exp1337` writes a terminal environment audit with `environment_ready`.
- `exp1338` closes `.103` stale skeleton/gate state into a precise carry-forward.
- `exp1339` produces `dynamic_grammar_ready=true` or a terminal grammar blocker.
- `exp1340` reaches `certificate_parse_rate >= 0.75` or retires the current
  AR certificate-tail recovery branch with evidence.
- `exp1341` separates certificate failures by failure mechanism and forbids a
  universal hallucination-detector claim.
- `exp1342` executes semantic validators on parsed certificates and preserves
  UNKNOWN states.
- `exp1343` reports false-acceptance risk before claiming verifier-call savings.
- `exp1344` satisfies the continuous self-learning requirement with
  non-forgetting and accepted-violation accounting.
- `exp1345` and `exp1346` run only if structured gates pass.
- `exp1347` and `exp1348` improve hardware portability evidence without
  unverified hardware claims.
- `exp1349` maps external EBT/Kona positioning to Carnot's actual local
  obligations.
- `exp1350` reconciles the milestone state and names `.105` carry-forwards with
  prior-failure hygiene.

## Decentralization Implication

The `.104` plan preserves Carnot's local-first posture. The headline path uses
local open GGUF models and deterministic local validators; hardware work is
simulation/accounting unless real hardware executes; external systems such as
Kona, Extropic, XGrammar, and ChopChop are used as design references, not
closed dependencies in the Carnot core.
