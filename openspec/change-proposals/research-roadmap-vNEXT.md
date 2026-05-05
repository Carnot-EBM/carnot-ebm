# Research Roadmap vNEXT: Milestone 2026.04.105

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.104 Environment Gate Recovery + Triggered Semantic Certificates + Verifier-Cost Self-Learning
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .104 Proved

Milestone 2026.04.104 was a useful recovery milestone, but it did not produce
the missing SOTA certificate evidence. The next milestone must not treat the
absence of `exp1340` as a negative result about trigger-switched certificates;
it is a gate-order / artifact-missing result that needs a terminal replacement
artifact.

| Track | Evidence | Finding |
|---|---|---|
| Environment gate | `exp1337` | `environment_ready=true`; stale `.103` artifacts were classified and disk/pretest state was made explicit. |
| .103 carry-forward | `exp1338` | `exp1325` was correctly classified as a stale environment failure, not scientific evidence against certificate recovery. |
| Dynamic grammar | `exp1339` | `dynamic_grammar_ready=true` via local pure-Python TagDispatch dry-run; XGrammar itself was absent, but the branch model is ready for a guarded SOTA attempt. |
| SOTA certificate run | `exp1340` | Missing. The parse gate was never measured in `.104`, so downstream semantic/DVI gates must remain closed until a terminal certificate artifact exists. |
| Failure taxonomy | `exp1341` | Diagnostic split completed from local cases, with `universal_detector_claim_allowed=false`. |
| Semantic validator / scheduler | `exp1342`, `exp1343` | Semantic validator did not run because the certificate artifact was missing; scheduler blocked correctly. |
| Continuous self-learning | `exp1344` | Replay-only policy was positive: `self_learning_delta_overall=1.596429`, `dvi_ready=true`, but `headline_result_allowed=false` without fresh certificate cases. |
| Hardware / external parity | `exp1347`, `exp1348`, `exp1349` | THRML unavailable locally, p-bit dual-BRAM packet updated, no hardware claims, and no external Kona/Extropic dependency claims. |
| Retro | `exp1350` | `.104` met 9 of 12 criteria and named six carry-forward tasks, led by terminal `exp1340` replacement and gated semantic/DVI work. |

The natural `.105` shape is therefore: close the `.104` handoff, preflight the
certificate grammar so it cannot truncate silently, run one terminal SOTA
certificate experiment, decompose failures by formal skill, and only then
advance semantic repair, DVI, and GRPO.

## Research Signals Added Before Planning

The post-.104 sweep added the following 2025-2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2602.06533`, LogicSkills: splits formal reasoning into
  symbolization, countermodel construction, and validity assessment, all
  solver-verified with Z3.
- `arXiv:2602.18095`, Logitext: treats natural-language text constraints as an
  SMT theory, preserving partial formalization and coverage.
- `arXiv:2601.20055`, VERGE: uses semantic routing, formal equivalence, and
  Minimal Correction Subsets for repair-localized verifier feedback.
- OpenReview `lrc2xSoh9b`, TruncProof, plus the 2026-05-04 MLC XGrammar-2
  runtime post: certificate grammars need max-token completion checks before
  expensive local SOTA generation.
- `arXiv:2603.03297` and `arXiv:2505.19475`: test-time self-reflection and
  verifier-driven sample selection provide a safer path from replay-positive
  self-learning to fresh verifier-selected updates.
- `arXiv:2506.00269`, p-dits: multi-valued probabilistic variables map Carnot's
  SAT/UNSAT/UNKNOWN/repair certificate states more naturally than binary
  one-hot p-bits.
- `arXiv:2512.15605` v3 and EBT's ICLR 2026 oral status: Carnot should keep
  Phase-3 claims grounded in local verifier evidence rather than broad EBT
  analogy.

## Three Biggest Gaps

1. **Terminal certificate evidence gap.** `.104` proved the environment and
   grammar dry-run were ready, but `exp1340` did not leave a terminal artifact.
   Until a SOTA certificate run records parse/truthfulness/UNKNOWN metrics or a
   terminal blocker, semantic validator and DVI work remain legitimately closed.

2. **Skill-localized semantic repair gap.** Carnot still has a single parse
   gate standing in for multiple failure modes. LogicSkills, Logitext, and
   VERGE point to the missing decomposition: symbolization versus validity
   versus countermodel, fully formal versus partial natural-language
   constraints, and MCS-localized repair hints.

3. **Headline self-learning gap.** `exp1344` is positive replay evidence, but
   FR-11 still needs fresh verifier-selected samples with non-forgetting and
   accepted-violation controls. DVI and GRPO must stay gated until fresh
   certificate and semantic evidence exists.

## Architecture Target

```text
Phase 0: handoff closure and certificate completion preflight
  exp1351 .104 carry-forward and artifact integrity audit
      |
      v
  exp1352 TruncProof/XGrammar completion-budget preflight
      |
      v
Phase 1: terminal SOTA certificate and semantic repair
  exp1353 trigger-switched certificate v7 on mandated local GGUF
      |
      +--> exp1354 LogicSkills skill split
      |
      v
  exp1355 Logitext/NSVIF partial SMT validator
      |
      v
  exp1356 VERGE MCS repair localization
      |
      v
  exp1357 margin-aware Cactus/BEAVER scheduler v2

Phase 2: verifier-governed self-learning
  exp1358 verifier-selected memory update with replay fallback
      |
      +--> exp1359 DVI certificate-tail v4
              |
              v
          exp1360 GRPO/VPRM v14 micro-audit

Phase 3: hardware/claim boundary and closeout
  exp1361 p-dit/p-int certificate-state hardware mapping
  exp1362 publication hold + EBT/ARM/Kona claim boundary
  exp1363 milestone retro and .106 carry-forward
```

## Phase 0: Handoff Closure and Certificate Completion Preflight

Goal: prevent another missing `exp1340`-class artifact before any GPU-heavy SOTA
work runs.

- `exp1351-104-carryforward-artifact-integrity-audit`: read `.104` artifacts,
  conductor log, and retro; produce a terminal integrity artifact that states
  which gates are open, which are closed, and which prior failures must be
  attached to `.105` tasks.
- `exp1352-truncproof-xgrammar-certificate-completion-preflight`: perform a
  local grammar/token-budget preflight for SAT/UNSAT/UNKNOWN/repair
  certificate states. The SOTA run is allowed only if the grammar can complete
  inside the configured max-token budget and dynamic branch dispatch remains
  valid.

Success bar: `sota_run_allowed=true`, or a terminal blocker that prevents
`exp1353` from running.

## Phase 1: Terminal SOTA Certificate and Semantic Repair

Goal: replace the missing `.104` SOTA certificate branch with one terminal
artifact, then decompose and validate what it produced.

- `exp1353-triggered-certificate-v7-truncproof-sota`: run the mandated local
  SOTA GGUF pair through trigger-before-constrain plus dynamic grammar and
  completion-budget checks. This task must write a terminal artifact even if it
  blocks.
- `exp1354-logicskills-certificate-skill-split`: split certificate results into
  symbolization, countermodel, and validity-assessment rates instead of hiding
  all failure behind parse rate.
- `exp1355-logitext-nsvif-partial-smt-validator`: when parse evidence clears
  the gate, route fully formal claims to local SMT and partial claims to
  natural-language text constraints while preserving UNKNOWN.
- `exp1356-verge-mcs-repair-localization`: use MCS-style localization to turn
  semantic rejects into minimal repair hints.
- `exp1357-margin-aware-cactus-beaver-scheduler-v2`: rerun scheduler work only
  after semantic and repair-localization evidence exists.

Success bar: either `certificate_parse_rate >= 0.75` plus semantic validation,
or a terminal retirement of the current trigger-switched certificate branch with
evidence about the remaining blocker.

## Phase 2: Verifier-Governed Continuous Self-Learning

Goal: satisfy the continuous self-learning requirement without letting replay
success become an unsupported headline claim.

- `exp1358-continuous-self-learning-verifier-selected-memory`: mandatory
  FR-11 task. It always runs with replay fallback from `exp1344`, and upgrades
  to headline evidence only when fresh `.105` validated certificate cases exist.
- `exp1359-dvi-certificate-tail-v4-gated`: apply DVI only after parse,
  semantic, and non-forgetting gates pass.
- `exp1360-grpo-vprm-v14-gated`: run a bounded replay or micro-audit only if
  DVI claims lossless acceptance and the self-learning delta is positive.

Success bar: self-learning reports non-forgetting, memory regression, and
accepted-violation deltas. DVI/GRPO remain closed unless gates pass.

## Phase 3: Hardware/Claim Boundary and Closeout

Goal: keep Phase-2/Phase-3 architecture current without overclaiming.

- `exp1361-pdit-certificate-state-hardware-mapping`: map certificate states and
  memory variables to p-dits/p-ints, comparing against binary p-bit expansion.
  This is CPU-only mapping; no hardware claim is allowed.
- `exp1362-publication-hold-ebt-arm-kona-claim-boundary`: update the local
  claim boundary using `.105` results, EBT/ARM-EBM theory, and `exp1349`.
- `exp1363-milestone-105-retro-carryforward`: evaluate `.105` criteria, name
  `.106` carry-forwards with prior-failure hygiene, and keep the publication
  hold honest.

Success bar: hardware mapping improves future FPGA/TSU packet quality without
claiming execution, and the retro leaves no missing-artifact ambiguity.

## Dependency Graph

```text
exp1351 ---> exp1352 ---> exp1353 ---> exp1354
                            |
                            v
                         exp1355 ---> exp1356 ---> exp1357
                            |
exp1344 + exp1353/1355 ---> exp1358 ---> exp1359 ---> exp1360

exp1348 + p-dit refs -----> exp1361
exp1349 + EBT/ARM refs ---> exp1362
exp1351..exp1362 --------> exp1363
```

## Hardware Requirements

- Fresh LLM-bearing tasks must use local mandated GGUF models through
  `cached_sota_pair(gpu_indices=(0, 1))` or an equivalent helper. Every fresh
  LLM task must include at least one of:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`,
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- Legacy small models such as Qwen3.5-0.8B and gemma-4-E4B-it may only be
  CPU smoke tests with `headline_result_allowed=false`.
- `exp1353` is the only planned fresh SOTA generation task. It should use the
  two RTX 3090s when possible and record model IDs, quantization, GPU
  assignment, token budgets, completion-budget preflight status, parse metrics,
  and throughput.
- `exp1361` is hardware mapping only. It may not claim FPGA, KV260, p-dit ASIC,
  TSU, ROCm, analog, or Kona execution unless that runtime actually executes.

## Success Criteria

- `exp1351` writes a terminal `.104` carry-forward integrity audit.
- `exp1352` either allows the SOTA certificate run with completion-budget
  evidence or blocks it terminally.
- `exp1353` writes terminal certificate evidence; missing artifact is failure.
- `exp1354` reports skill-specific certificate failure rates.
- `exp1355` runs only when parse evidence clears the structured gate and
  reports UNKNOWN-preserving semantic validation.
- `exp1356` reports MCS-localized repair hints or a terminal repair blocker.
- `exp1357` reports false-acceptance risk before claiming verifier-call savings.
- `exp1358` satisfies the mandatory continuous self-learning requirement and
  separates replay-only from headline evidence.
- `exp1359` and `exp1360` run only if structured gates pass.
- `exp1361` produces p-dit/p-int mapping evidence without hardware claims.
- `exp1362` keeps publication claims aligned with local evidence.
- `exp1363` reconciles the milestone and names `.106` carry-forwards with
  prior-failure hygiene.

## Decentralization Implication

The `.105` plan preserves Carnot's local-first posture. The only fresh LLM
headline path uses local open GGUF models; semantic validators and self-learning
gates are local deterministic/replay code; hardware work is mapping-only unless
real hardware executes; and external systems such as EBT, ARM-EBM, Kona,
Extropic, XGrammar, TruncProof, and VERGE are design references, not closed
dependencies in Carnot core.
