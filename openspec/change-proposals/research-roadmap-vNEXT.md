# Research Roadmap vNEXT: Milestone 2026.05.280

**Title:** Intent-Preserving Repair + Solver Feedback + Readback-Grounded Self-Learning

**Planned:** 2026-05-24

**Previous milestone:** 2026.05.279

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.279 Proved

Milestone `.279` completed all scheduled tasks. The authoritative terminal
artifact is `results/experiment_2974_capstone_v279.json`, with
`paper_ready=false`.

- Structured repair is real infrastructure but not yet useful. `exp2963`
  produced a DCCD structured-repair protocol, but `exp2964` showed the live
  repair rerun regressed: `pass@1_delta=-0.2`, `pass@k_delta=-0.3`, and
  `schema_failure_rate_delta=0.95`.
- BEAVER-style bounded certificate fields landed in `exp2965`, but only as a
  bounded audit. No full BEAVER probability-bound claim exists.
- Logic exact-frontier materialization worked in `exp2966`, and SOTA local
  NL-to-Z3 improved over `.278`, but `exp2967` still missed promotion gates:
  parseability `0.25`, Z3 execution `0.208333`, and solver-verified accuracy
  `0.208333`.
- Interwhen-style partial monitors landed as a deterministic harness in
  `exp2968`, but stayed pilot-only rather than a full streaming verification
  claim.
- FR-11 self-learning became non-tautological in `exp2969`, but the capstone
  still flagged it because independent metric slices were not strong enough for
  a headline autonomous self-learning claim.
- The KAN forgetting guard in `exp2970` was clean and bounded to fixture-level
  memory evidence.
- GateMate board contact and flash/output-hash evidence landed in `exp2971`
  and `exp2972`, but there is still no readback-backed smoke vector or sampler
  claim.
- Matrix v13 (`exp2973`) aggregated the state as complete but adversarially
  flagged: clean `26`, flagged `14`, blocked `5`, pilot-only `4`.

## Three Biggest Gaps

### Gap 1: Structured Repair Currently Damages Pass Rate

The largest gap is not missing constrained decoding; it is harmful constrained
decoding. `.279` showed that schema-driven DCCD increased failures and reduced
pass rate. `.280` must test intent-preserving, trace-aware repair inspired by
AdapTrack, TraceCoder, Thinking Before Constraining, and backtracking
constrained-decoding work. The target is not prettier JSON. The target is a
positive code-repair delta with no schema/syntax regression and no false-accept
increase.

### Gap 2: Solver-Grounded Reasoning Needs Feedback, Not More Prompts

The local GGUF formalization row now has a clean exact-verifier frontier, but
the model outputs are still mostly unparseable or solver-wrong. `.280` should
add solver feedback objects, MCS/MUS-style unsat diagnostics, partial-monitor
promotion criteria, and first-step/prefix failure telemetry. Z3 remains the
authority; LLM output remains a proposal.

### Gap 3: FR-11 and Hardware Need Independent Evidence

FR-11 needs independent metric slices that cannot be satisfied by the same
utility function used for update selection. GateMate needs readback or a passed
smoke vector before Carnot can claim any sampler-facing hardware evidence.
`.280` should connect these carefully: a continuous self-learning task may use
repair traces as memory, but only if held-out utility, negative controls, and
forgetting guards remain separate.

## New Research Integrated

The 2026-05-24 post-`.279` sweep appended these items to
`research-references.md`:

- **AdapTrack constrained decoding without output-intent distortion (ICSE
  2026):** motivates intent-preserving constrained repair rather than stricter
  schema masks.
- **TraceCoder (arXiv:2602.06875):** motivates execution traces as repair
  evidence and reusable self-learning memories.
- **Thinking Before Constraining (arXiv:2601.07525):** supports draft-first,
  constrain-later protocols for code and solver formalization.
- **Constrained Decoding Diffusion LLMs (arXiv:2604.26139):** motivates
  backtracking/repair-style constrained generation instead of token-only masks.
- **Taming Imperfect Process Verifiers (OpenReview ICLR 2026):** reinforces
  rollback and negative-control discipline for verifier-guided learning.
- **ARM-as-EBM citation watch:** reinforces latent-loop, ontology-constrained,
  and graph-energy diagnostics without triggering native EBT training.
- **False first steps / first-token telemetry:** motivates early-prefix
  diagnostics as candidate triage, not as a verifier.
- **FPGA SSQA dual-BRAM and memory-efficient SSA:** motivates readback,
  register maps, smoke vectors, and memory accounting for GateMate/KV260.

## Architecture Snapshot

```text
        +---------------------------------------------------------+
        | Phase A: intent-preserving repair                       |
        |                                                         |
        | exp2975 archive/activate                               |
        | exp2976 AdapTrack + TraceCoder protocol                 |
        | exp2977 gated SOTA repair rerun                         |
        | exp2978 first-step / semantic-energy telemetry          |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase B: solver feedback and monitor promotion          |
        |                                                         |
        | exp2979 solver feedback schema + MCS/MUS frontier       |
        | exp2980 gated SOTA solver formalization                 |
        | exp2981 partial monitor promotion                       |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase C: continuous self-learning evidence              |
        |                                                         |
        | exp2982 FR-11 independent metric utility gate           |
        | exp2983 trace-to-skill repair memory pilot              |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase D: hardware readback and closeout                 |
        |                                                         |
        | exp2984 GateMate readback + smoke vector                |
        | exp2985 SSQA dual-BRAM register-map plan                |
        | exp2986 cross-corpus matrix v14                         |
        | exp2987 capstone .280                                   |
        +---------------------------------------------------------+
```

## Phase Structure

### Phase A: Intent-Preserving Repair

- `exp2975` archives `.279` and stages `.280` without touching
  `research-roadmap.yaml`.
- `exp2976` converts the failed DCCD row into a pre-registered
  intent-preserving repair protocol. It must include semantic draft retention,
  backtracking constraints, execution traces, schema gates, and false-accept
  audit fields.
- `exp2977` is the live local SOTA repair rerun. It must use the mandated GGUF
  models through `cached_sota_pair()` when available, with legacy tiny models
  allowed only as CPU smoke tests. Promotion requires positive pass-rate deltas
  without schema/syntax or false-accept regression.
- `exp2978` adds first-step, prefix, and semantic-energy telemetry for repair
  candidates. It is triage evidence only; it may not become a verifier claim.

### Phase B: Solver Feedback and Monitor Promotion

- `exp2979` adds solver-feedback objects and MCS/MUS-style diagnostics to the
  existing exact-verifier frontier.
- `exp2980` reruns local SOTA NL-to-Z3 formalization with feedback-aware
  prompting and deterministic Z3 authority.
- `exp2981` promotes the Interwhen partial monitor harness only if it measures
  event coverage and failure-localization. It must explicitly keep
  `full_streaming_verification_claim=false` unless coverage supports more.

### Phase C: Continuous Self-Learning Evidence

- `exp2982` is the required continuous self-learning experiment. It reruns the
  FR-11 gate with independent metrics, random replay, frozen baseline, negative
  controls, and forgetting checks.
- `exp2983` uses execution traces from repair/formalization as candidate skill
  memories and tests whether those memories improve held-out tasks without
  reusing the same update-selection utility.

### Phase D: Hardware Readback and Closeout

- `exp2984` tries to convert GateMate board contact into readback-backed smoke
  evidence. It may report flash, hash, readback, and smoke-vector status only.
  No speedup, Boltzmann, thermodynamic, or sampler claim is allowed unless the
  artifact records sample-level evidence.
- `exp2985` converts FPGA sampling references into a projection-only
  register-map and memory-layout plan for GateMate/KV260.
- `exp2986` rebuilds the cross-corpus matrix v14 with `.280` repair, solver,
  self-learning, monitor, and hardware rows.
- `exp2987` closes `.280`, classifies clean/flagged/blocked rows, and states
  whether any claim is paper-ready.

## Dependency Graph

```text
exp2975

exp2976
  -> exp2977
  -> exp2983

exp2977
  -> exp2978

exp2979
  -> exp2980
  -> exp2981

exp2982
  -> exp2986

exp2984
exp2985

exp2977
exp2978
exp2980
exp2981
exp2982
exp2983
exp2984
exp2985
  -> exp2986

all artifacts
  -> exp2987
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2977` gates on
  `exp2976.intent_preserving_repair_protocol_ready == true`.
- `exp2980` gates on `exp2979.mcs_feedback_schema_ready == true`.
- `exp2981` gates on `exp2979.frontier_upgrade_ready == true`.
- `exp2983` gates on `exp2976.trace_execution_plan_ready == true`.
- `exp2986` gates on
  `exp2982.fr11_independent_metrics_evaluated == true`.
- `exp2987` is intentionally ungated so the milestone can close honestly even
  if a branch is blocked or skipped.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- Mandatory local GGUF headline models for any new LLM experiment:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Experiments that need live inference must call `cached_sota_pair()` first and
  record exact `models_used`. Tiny legacy models may appear only as
  non-headline CPU smoke tests.

Required for GateMate tasks:

- GateMate A1-EVB-2M USB-attached board.
- The `.278/.279` n=16 Ising tile bitstream, flash tooling, board identifier,
  output hash path, and any available readback command.
- No speedup/sampler claim until a passed smoke vector and sample-level timing
  or readback evidence exist.

Optional but useful:

- KV260 SSH/UIO setup for later register-map transfer.
- PolarFire SSH host for cross-FPGA accounting only.
- Extropic THRML simulator for future readiness checks, not hardware claims.

## Acceptance Criteria

The milestone is successful if it produces one of these outcomes honestly:

- A clean repair row: `exp2977` reports `repair_rerun_clean=true`,
  `n_tasks>=20`, positive pass-rate deltas, no schema/syntax regression, and no
  false-accept increase.
- A clean solver row: `exp2980` reports parseability and Z3 execution at least
  `0.50`, solver-verified accuracy at least `0.40`, and no tautological answer
  matching.
- A clean FR-11 row: `exp2982` reports independent held-out improvement over
  random replay, no negative-control improvement, and passed forgetting guards.
- A clean hardware row: `exp2984` records readback or a passed smoke vector
  without overclaiming sampling, speedup, or thermodynamic behavior.

If none of these clear, `.280` still succeeds operationally if `exp2987`
classifies the failure modes and retires repeated dead ends with honest
`prior_failures` mechanics.

## Failed-Experiment Rerun Compliance

This milestone intentionally revisits several failed or flagged scopes from
`.279`. Every matching task in `research-roadmap-next.yaml` includes a
`prior_failures` block with all required fields, including
`retire_if_same_verdict: true`.

Expected repeats:

- DCCD repair: `exp2963`, `exp2964`, and earlier small-N repair rows.
- Solver formalization: `exp2966`, `exp2967`, and related `.278` solver rows.
- Partial monitors: `exp2968`.
- FR-11 self-learning: `exp2969` and earlier utility-gated replay rows.
- GateMate hardware: `exp2972` and prior board/bitstream blockers.
- Matrix/capstone aggregation: `exp2973`, `exp2974`.

## Out of Scope

- Native EBT training or claims of EBT-equivalent inference.
- Extropic TSU hardware claims.
- Kona/Aleph-equivalent claims.
- Full BEAVER probability-bound implementation.
- Speedup claims from GateMate, KV260, PolarFire, or any FPGA unless the
  artifact records passed smoke vectors and sample-level timing evidence.
