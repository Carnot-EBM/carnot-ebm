# Research Roadmap vNEXT: Milestone 2026.04.101

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.100 SOTA Certificate Recovery + Verifier-Feedback Self-Learning + Continuous Repair Bridge
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .100 Proved

Milestone 2026.04.100 met 5 of 14 planned criteria. That score is important
because the failures were mostly activation and dependency failures, not negative
research evidence:

- `exp1283` selected `llama.cpp` GBNF as the local certificate grammar backend.
  Grammar generation is therefore no longer speculative.
- `exp1288` proved verifier-feedback replay can improve acceptance rate
  (`dvi_acceptance_delta=0.357143` and `memory_update_written=true`), but the
  result stayed non-headline because SOTA certificate data never materialized.
- `exp1291` showed HardNet++ nonlinear repair is viable over SnareNet
  (`hardnetpp_delta_over_snarenet=1.2207`).
- `exp1292` showed the DSP feasibility channel is predictive but marginal
  (`feasibility_channel_auc=0.6605`) and should be used as a stop/continue
  signal, not as the only repair operator.
- `exp1295` wrote the honest retro and identified the avoidable waste pattern:
  missing `prior_failures` metadata caused repeated DOOMED_RERUN_BLOCKs in
  `exp1282`, `exp1293`, and `exp1294`, cascading into SOTA certificate, semantic
  routing, Cactus, GRPO, and publication gates.

The next milestone therefore starts with activation hygiene. It should not spend
GPU or model time until the roadmap itself proves that prior-failure coverage,
structured gates, and cache/provenance readiness are dispatchable.

## Current Research Signals Added Before Planning

The 2026-05-05 planning sweep added source-backed items to
`research-references.md` before this roadmap was designed:

- FALCON (`arXiv:2602.01090`) motivates a certificate path that combines
  grammar-constrained decoding, semantic feasibility repair, and adaptive
  Best-of-N sampling.
- Attention Meets Reachability (`arXiv:2603.05540`) makes grammar state count,
  ambiguity cost, and grammar-induced latency first-class certificate metrics.
- Semantic Probabilistic Control of Language Models (`arXiv:2505.01954`,
  OpenReview SPIGM 2025) motivates verifier-feature routing rather than
  syntax-only routing.
- QueryBandits (`arXiv:2602.20332`) and Neural Garbage Collection
  (`arXiv:2604.18002`) sharpen continuous self-learning: memory policy must
  include online selection, demotion, and expiry, not promotion-only ledgers.
- Optimal KAN abstractions (`arXiv:2602.06737`) provide a future path from
  nonlinear repair to PWA/MILP-verifiable bounded abstractions.
- Infeasibility-aware LLMs for combinatorial optimization (`arXiv:2604.01455`)
  require certificate schemas to preserve infeasible and unknown states.
- Recent p-bit, Extropic TSU, and Logical Kona updates reinforce the energy and
  hardware direction, but do not unblock FPGA/TSU work locally in this milestone.

## Three Biggest Gaps

1. **Activation and prior-failure discipline gap.** `.100` wasted multiple
   conductor attempts because carry-forward tasks lacked explicit
   `prior_failures`. The PRD cannot be advanced by experiments that never
   activate. `.101` must prove `research-roadmap-next.yaml` is lintable and
   gate-auditable before SOTA work begins.

2. **Headline local SOTA certificate gap.** The mandated local GGUF models still
   have not produced a headline-eligible certificate parse-rate, answer-stability
   result, or semantic routing corpus. This remains the largest gap between
   current state and FR-12 verifiable reasoning.

3. **Continuous self-learning control gap.** `.100` proved a positive DVI
   acceptance delta and wrote memory, but did not emit the skill graph and did
   not improve `self_learning_delta_overall`. FR-11 needs closed-loop online
   learning with promotion, demotion, routing, and measured regret/violation
   deltas.

Continuous repair is the fourth gap. `.100` made it promising; `.101` should
turn HardNet++ and DSP diagnostics into an explicit stop/continue policy and
connect that policy back to Phase-3 energy semantics.

## Architecture Target

```text
Phase 0: activation and local cache readiness
  roadmap prior-failure + gate audit
      |
      v
  SOTA GGUF cache/provenance preflight v2
      |
      +--------------------------------------+
      |                                      |
      v                                      v
Phase 1: SOTA certificate path          Phase 3 bridge carry-forward
  answer-stability on mandated GGUFs       EBT/ARM/EBM-CoT/Kona audit
      |
      v
  triggered <CARNOT_CERT> extraction v3
  llama.cpp GBNF + FALCON repair + adaptive Best-of-N
      |
      +---------------------------+
      |                           |
      v                           v
  semantic verifier routing       safe-prefix/Cactus acceptance
  SConE/FALCON/infeasible states  Token-Guard/HoVer risk filters

Phase 2: continuous self-learning
  exp1288 verifier-feedback memory
      |
      v
  skill graph promotion/demotion v2
      |
      v
  QueryBandits/NGC online memory policy
      |
      v
  GRPO/VPRM v10 only if SOTA cert + online-learning gates pass

Phase 3: repair, bridge, publication state, retro
  HardNet++ + DSP feasibility stop policy
      |
      +--> EBT/ARM/EBM-CoT + Extropic/Kona bridge audit v2
      +--> arXiv v10 hold/receipt terminal artifact
      +--> milestone .101 retro
```

## Phase 0: Activation Hygiene and SOTA Readiness

Goal: eliminate the avoidable `.100` waste before any expensive work runs.

- `exp1296-prior-failures-activation-audit`: validate
  `research-roadmap-next.yaml` with the existing prior-failure and gate audit
  scripts, verify that every carry-forward task has explicit `prior_failures`,
  and write a terminal artifact with `prior_failures_coverage_ok`.
- `exp1297-sota-gguf-cache-provenance-preflight-v2`: rerun the cache/provenance
  preflight with explicit priors for `exp1282`, `exp1271`, `exp785`, and
  `exp811`, using `cached_sota_pair()` and recording all mandated SOTA GGUF
  model specs.

Success bar: downstream SOTA tasks are gated on a passing prior-failure audit
and `cached_sota_ready=true`. If either fails, the milestone still writes a
usable blocker instead of repeating `.100`.

## Phase 1: SOTA Certificates and Semantic Acceptance

Goal: finally measure the local SOTA certificate path and only then run routing
and constrained acceptance.

- `exp1298-sota-answer-stability-falcon-audit`: use the mandated SOTA GGUF
  pair to measure answer stability, cross-model disagreement, infeasible/unknown
  detection, and FALCON-style repair opportunity on a small verifier benchmark.
- `exp1299-triggered-certificate-extraction-v3`: use the `.100` grammar backend
  plus FALCON repair and adaptive Best-of-N sampling to compare raw-trigger,
  GBNF, repaired, and adaptive certificate paths. It must measure parse rate,
  truthfulness, grammar cost, and repair success.
- `exp1300-semantic-routing-v2`: route parsed certificate claims to verifiers
  using syntax, semantic verifier features, infeasible/unknown states, and
  minimal correction diagnostics.
- `exp1301-safe-prefix-cactus-acceptance-v3`: run HoVer/Token-Guard/Cactus-style
  low-risk acceptance only if certificate parse rate and semantic routing gates
  open.

Success bar: `exp1299.certificate_parse_rate >= 0.8` and
`headline_result_allowed=true`, or a precise blocker explains why the local SOTA
path remains closed.

## Phase 2: Continuous Self-Learning

Goal: make FR-11 measurable with a closed loop that can promote and demote
memory.

- `exp1302-skill-graph-promotion-demotion-v2`: recover the `.100` missing skill
  graph task, gated on `exp1288.memory_update_written`, with explicit promoted,
  demoted, expired, and replay-backed memory entries.
- `exp1303-querybandits-ngc-online-memory-policy`: run the mandatory continuous
  self-learning experiment. It should treat replay/rewrite/abstain/demote
  choices as arms, measure regret and accepted-violation deltas, and report
  `self_learning_delta_overall`.
- `exp1304-grpo-vprm-v10-sota-gated`: run only if SOTA certificates are
  headline-eligible and the online memory policy improves self-learning. This
  keeps expensive learning off the critical path unless the evidence is present.

Success bar: at least one artifact reports a positive online-learning or
memory-routing delta and writes terminal promotion/demotion evidence. If the
SOTA gate stays closed, self-learning still runs on replay data and marks
headline eligibility honestly.

## Phase 3: Repair, Energy Bridge, Publication State, Retro

Goal: turn `.100` repair positives into policy and close blocked carry-forwards.

- `exp1305-hardnetpp-dsp-feasibility-stop-policy`: combine HardNet++ and DSP
  findings into a stop/continue benchmark policy with residual nonlinear and
  KAN/PWA abstraction notes.
- `exp1306-ebt-arm-ebm-cot-energy-bridge-audit-v2`: rerun the blocked energy
  bridge with explicit priors for `exp1293` and `exp458`, incorporating EBT
  citation signals, ARM-EBM, EBM-CoT, Extropic TSU, p-bit, and Kona context.
- `exp1307-arxiv-v10-hold-receipt-v2`: write a terminal publication artifact
  with explicit priors for `exp1294`, `exp1127`, `exp1139`, and `exp1153`.
  It must not attempt credentialed arXiv submission while the known operator
  hold remains in force.
- `exp1308-milestone-retro-101`: mechanically score the milestone, name
  carry-forwards, and reconcile planning/ops docs.

Success bar: repair policy fields are measurable, the energy bridge no longer
blocks on missing priors, publication state is explicit, and the retro is honest
about gates that skipped work.

## Dependency Graph

```text
exp1296 ---> exp1297 ---> exp1298 ---> exp1299 ---> exp1300 ---> exp1301
                         |              |
                         |              +--------------------+
                         |                                   |
exp1288 ----------------> exp1302 ---> exp1303 ------------> exp1304

exp1291/exp1292 context ----------------> exp1305
exp1293/exp458 priors ------------------> exp1306
exp1294/1127/1139/1153 priors ----------> exp1307

exp1296..exp1307 -----------------------> exp1308
```

Structured conductor gates:

- `exp1297` gates on `exp1296.prior_failures_coverage_ok == true`.
- `exp1298` gates on `exp1297.cached_sota_ready == true`.
- `exp1299` gates on `exp1297.cached_sota_ready == true`,
  `exp1296.exp1283_grammar_backend_available == true`, and
  `exp1298.answer_stability_score >= 0.6`.
- `exp1300` gates on `exp1299.certificate_parse_rate >= 0.8`.
- `exp1301` gates on `exp1299.certificate_parse_rate >= 0.8` and
  `exp1300.semantic_routing_coverage >= 0.5`.
- `exp1302` gates on `exp1296.exp1288_memory_update_written == true`.
- `exp1303` gates on `exp1302.skill_graph_candidate_count > 0`.
- `exp1304` gates on `exp1299.headline_result_allowed == true` and
  `exp1303.self_learning_delta_overall > 0.0`.

## Hardware Requirements

Minimum CPU-only path:

- `exp1296`, `exp1302`, `exp1303`, `exp1305`, `exp1306`, `exp1307`, and
  `exp1308` can run without GPUs.
- `exp1297` is a cache/provenance preflight and should not download models
  unless the existing resolver already performs a safe cache lookup.

Required for headline LLM results:

- Mandated local SOTA GGUFs through `cached_sota_pair()`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Prefer the dual RTX 3090 CUDA path for `exp1298`, `exp1299`, `exp1301`, and
  any `exp1304` run. Legacy small models may only be smoke tests with
  `headline_result_allowed=false`.

Not required in `.101`:

- KV260/Vivado FPGA synthesis remains human-blocked.
- AMD XDNA/NPU work remains human-install blocked.
- Extropic TSU/Z1 access remains strategic context, not a local dependency.

## Decentralization Implications

The milestone remains local-first. LLM-bearing experiments must use local
open-weight GGUF `MODEL_SPECS` and record exact model IDs/paths. Closed vendor
LLMs are not part of the scientific result path. Hardware claims stay behind
sampler and energy abstractions until local FPGA/TSU prerequisites change.

## Milestone Success Criteria

1. `exp1296` reports `prior_failures_coverage_ok=true` and gate audit pass, or
   names every remaining activation blocker.
2. `exp1297` records SOTA GGUF cache/provenance readiness or exact missing
   model blockers.
3. `exp1298` measures SOTA answer stability, cross-model disagreement,
   infeasible/unknown handling, and FALCON-style repair opportunity.
4. `exp1299` produces headline-eligible SOTA certificate parse/truthfulness
   metrics or a precise blocker, with grammar cost and repair metrics.
5. `exp1300` writes semantic routing coverage and verifier-feature deltas.
6. `exp1301` measures safe-prefix/Cactus acceptance when parse/routing gates
   open.
7. `exp1302` emits skill graph promotion, demotion, expiry, and replay evidence.
8. `exp1303` satisfies the continuous self-learning mandate with online policy
   regret and self-learning/violation deltas.
9. `exp1304` runs GRPO/VPRM v10 only when the SOTA and self-learning gates pass.
10. `exp1305` writes a HardNet++/DSP feasibility stop-policy artifact.
11. `exp1306` completes the EBT/ARM/EBM-CoT bridge audit without prior-failure
    blockage.
12. `exp1307` records publication receipt or explicit hold/blocker without
    attempting credentialed submission.
13. `exp1308` completes the `.101` retrospective and carry-forward list.

## Key Planning Decisions

- The milestone sequence increments from `2026.04.100` to `2026.04.101`.
- The task count is 13 across four phases.
- All tasks use `agent_type: codex` and `model: gpt-5.5` per current
  `CLAUDE.md` guidance unless a future operator overrides routing.
- LLM-bearing tasks include explicit mandated SOTA GGUF requirements and the
  `cached_sota_pair()` pattern.
- Carry-forward tasks include explicit `prior_failures`; the first experiment
  validates this before downstream work.
- No task modifies `research-roadmap.yaml`.
- No task modifies `scripts/research_conductor.py`.
