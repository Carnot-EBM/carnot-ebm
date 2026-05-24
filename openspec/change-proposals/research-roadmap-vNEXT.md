# Research Roadmap vNEXT: Milestone 2026.05.281

**Title:** SOTA Cache Recovery + Verifier-Backed Repair + Provenance-Grounded Self-Learning

**Planned:** 2026-05-24

**Previous milestone:** 2026.05.280

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.280 Proved

Milestone `.280` completed the planned experiment set through
`results/experiment_2987_capstone_v280.json`. The capstone verdict is
`complete: milestone_280_capstone; paper_ready=false; clean=3; flagged=5;
blocked=2; missing=0`.

- **Archive activation is working.** `exp2975` archived `.279` cleanly, keeping
  roadmap activation mechanics usable for the next milestone.
- **Intent-preserving repair remains blocked, not disproven.** `exp2977`
  produced only CPU smoke evidence with `Qwen/Qwen3.5-0.8B`; the mandated SOTA
  cached local models were unavailable, so the headline repair gate stayed
  blocked.
- **Solver feedback is the strongest technical signal.** `exp2979` produced a
  deterministic MCS/MUS feedback frontier, and `exp2980` reached perfect local
  Z3 execution on six items, but the row stayed flagged because provenance,
  duration, and checksum evidence were too thin for a paper claim.
- **First-step and semantic-energy telemetry are useful triage signals.**
  `exp2978` showed strong proxy separation, but it remains calibration/triage
  evidence only, not a verifier.
- **FR-11 finally has a clean independent metric boundary.** `exp2982` cleared
  the non-identical-metric and negative-control gates; `exp2983` produced a
  trace-to-skill pilot, but it needs live verifier-backed carry-forward before
  becoming autonomous self-learning evidence.
- **GateMate still lacks host-visible output evidence.** `exp2984` detected the
  board and flashed, but produced no readback or smoke-vector output. `exp2985`
  gave a useful SSQA dual-BRAM register-map plan, still projection-only.
- **The cross-corpus matrix is honest.** `exp2986` and `exp2987` marked repair
  and GateMate as blocked, solver as flagged, and FR-11 independent metrics as
  clean. `.281` should repair those exact weak links rather than broaden scope.

## Three Biggest Gaps

### Gap 1: Local SOTA Evidence Is the Current Bottleneck

The PRD vision requires verifiable reasoning with modern local models, but the
latest repair row did not exercise the mandated GGUF SOTA models. Before any
new repair/solver headline, `.281` needs a cache/provenance preflight that
records model identity, size, checksum or cache path, inference substrate,
duration, and a real response for at least one mandated SOTA model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models remain allowed only as smoke tests.

### Gap 2: Repair Needs Verifier-Backed Hard Cases, Not Another Broad Rerun

`.279` showed over-constraining can reduce pass rate, and `.280` could not run
the SOTA repair test. The next attempt should first build a hard-code stress
manifest inspired by HARDTESTGEN and verifier-backed hard problem generation:
every item must have executable validity evidence, failing edge tests, and a
baseline wrong solution before it can enter the repair rerun.

### Gap 3: Self-Learning and Hardware Need Independent Evidence Loops

FR-11 now has clean independent metrics, but the next self-learning task must
use fresh verifier-backed traces and preserve negative controls/forgetting
guards. Hardware remains more basic: GateMate must expose host-visible smoke
output or readback before SSQA or sampler claims can advance. SSQA dual-BRAM
work should move from register-map plan to RTL/PnR/resource evidence without
claiming sampling speedup.

## New Research Integrated

The 2026-05-24 post-`.280` sweep appended these sources to
`research-references.md` and shaped the milestone:

- **Verifier-Backed Hard Problem Generation (arXiv:2605.06660):** use
  independent verifiers to generate valid, difficult stress items.
- **HardTests / HARDTESTGEN (arXiv:2505.24098):** synthesize stronger coding
  tests before measuring code repair.
- **ConstrainPrompt (OpenReview 2026):** compile prompt-defined constraints into
  executable validators.
- **STATIC constrained decoding (arXiv:2602.22647):** represent strict output
  spaces as accelerator-friendly sparse transition structures.
- **Attribution-Guided Decoding (OpenReview 2026):** treat prompt-region
  attribution as candidate-ranking telemetry, not a verifier.
- **GVPO and DeepVerifier (OpenReview/arXiv 2026):** combine outcome-verifiable
  and process-verifiable feedback for self-improvement.
- **KAN/Ising hardware updates:** keep local hardware claims bounded to
  synthesis/PnR/readback/sample evidence.

## Architecture Snapshot

```text
                  research-complete.yaml / exp2987 capstone
                                  |
                                  v
                    exp2988 archive .280 and activate .281
                                  |
                                  v
                  exp2989 mandated SOTA cache/provenance gate
                         /            |              \
                        /             |               \
                       v              v                v
        exp2990 hard repair set   exp2992 solver    exp2993 AquaForte/
        verifier manifest         provenance v2     BEAVER substrate fix
              |                    |        \             |
              v                    |         \            |
        exp2991 SOTA repair        |          v           |
        rerun on hard set          |     exp2994 prompt-to-validator
              \                    |     dialogue schema
               \                   |          |
                \                  v          v
                 ----------> exp2995 FR-11 verifier trace memory v2
                                  |
                                  v
                          exp2998 matrix v15
                                  |
                                  v
                          exp2999 capstone v281

        Hardware side branch:
          exp2996 GateMate host-visible readback/smoke
              |
              v
          exp2997 SSQA dual-BRAM RTL/PnR/resource report
              |
              v
          exp2998 matrix v15
```

## Phase Structure

### Phase A: Activation and Evidence Gates

1. **exp2988 - Archive `.280` and activate `.281`.**
   Archive `.280` results into `research-complete.yaml`, update operational
   ledgers as required by the existing archive pattern, and emit activation
   status for `.281`.

2. **exp2989 - Mandated SOTA GGUF cache/provenance preflight.**
   Run a bounded live preflight for the mandated local SOTA models. This task
   is intentionally upstream of all live LLM result claims so blocked cache
   state can skip downstream expensive tasks.

3. **exp2990 - Verifier-backed hard-code stress manifest.**
   Build a small but strict hard-code repair/evaluation set with executable
   tests, baseline-failing candidates, provenance, and edge-case rationale.

### Phase B: Claim Recovery for Repair and Solver

4. **exp2991 - Gated SOTA intent-preserving repair rerun.**
   Rerun the repair protocol only if SOTA cache and hard-set gates pass. The
   headline target is a positive pass@1/pass@k delta with no schema/syntax or
   false-accept regression.

5. **exp2992 - Solver formalization provenance reproduction.**
   Preserve `.280` solver gains while adding model/cache checksums, live
   durations, prompt hashes, Z3 transcript hashes, and a larger fixed item set.

6. **exp2993 - AquaForte/BEAVER honest substrate corrigendum.**
   Resolve the mandatory known issue around `exp2934` by separating true live
   LLM retry from enumerator-only fallback and relabeling substrate claims.

7. **exp2994 - Prompt-to-validator dialogue schema.**
   Convert ConstrainPrompt/STATIC ideas into a local exact-verifier protocol:
   prompts produce candidate constraints, constraints compile to validator
   trees, and Z3/runtime checks remain authority.

### Phase C: Continuous Self-Learning and Hardware Grounding

8. **exp2995 - FR-11 verifier-grounded trace memory v2.**
   Continuous self-learning task. Use fresh repair/solver traces as candidate
   memories, select updates by process evidence, and evaluate on independent
   held-out metrics with negative controls and forgetting checks.

9. **exp2996 - GateMate host-visible IO/readback smoke.**
   Add or prove absence of host-visible smoke output/readback. No sampler,
   speedup, Boltzmann, or thermalization claim is allowed.

10. **exp2997 - SSQA dual-BRAM RTL/PnR/resource report.**
    Convert the `.280` register map into bounded RTL/PnR evidence with smoke
    vector hooks and resource reporting, gated on hardware-readback evidence
    when available.

### Phase D: Synthesis and Go/No-Go

11. **exp2998 - Cross-corpus matrix v15.**
    Aggregate `.281` outcomes against paper-v6, PRD, OpenSpec, hardware, and
    self-learning claim boundaries.

12. **exp2999 - Milestone `.281` capstone.**
    Decide whether the milestone is paper-ready, identify blocked/flagged rows,
    and recommend the next exact milestone. No external publication action.

## Dependency Graph

```text
exp2988
  -> exp2989
       -> exp2991
       -> exp2992
       -> exp2993
  -> exp2990 -> exp2991
exp2992 -> exp2994 -> exp2995
exp2991 -----------^
exp2996 -> exp2997
exp2995 -> exp2998
exp2997 -> exp2998
exp2998 -> exp2999
```

Structured conductor gates are included for:

- `exp2991`: requires `exp2989.sota_headline_ready == true` and
  `exp2990.hard_code_stress_set_ready == true`.
- `exp2992`: requires `exp2989.sota_headline_ready == true`.
- `exp2993`: requires `exp2989.sota_headline_ready == true`.
- `exp2995`: requires `exp2992.solver_provenance_reproduced == true` and
  `exp2994.prompt_validator_protocol_ready == true`.
- `exp2997`: requires `exp2996.hardware_smoke_boundary_recorded == true`.
- `exp2999`: requires `exp2998.matrix_v15_ready == true`.

`exp2998` intentionally has no hard gate: the matrix must still run when
upstream rows are blocked, flagged, or skipped so the capstone can report the
true milestone state.

## Hardware Requirements

- **Dual RTX 3090 CUDA host:** required for live local SOTA GGUF preflight and
  headline LLM experiments. If the cache cannot produce at least one mandated
  SOTA model transcript, downstream live-result tasks must emit terminal
  blocked artifacts instead of substituting small models.
- **CPU-only fallback:** permitted only for archive, schema/protocol work, hard
  manifest construction, and explicit smoke tests. CPU smoke evidence cannot
  become a headline SOTA result.
- **GateMate:** required for `exp2996`; the only acceptable advancement is
  host-visible readback or a smoke-vector transcript. Flash-only evidence stays
  blocked for sampler-facing claims.
- **SSQA/GateMate/KV260 context:** `exp2997` may produce RTL/PnR/resource
  evidence and register-map smoke hooks. It must not claim sampler speed,
  Boltzmann correctness, thermodynamic behavior, or FPGA acceleration without
  board-visible sample/timing evidence.
- **Extropic/THRML and Kona/Aleph:** architecture references only. No local
  access or performance claim is assumed.

## Acceptance Criteria

- `research-references.md` contains the post-`.280` sweep before experiment
  design.
- `research-roadmap-next.yaml` activates `2026.05.281` and leaves
  `research-roadmap.yaml` unchanged.
- Every live LLM experiment includes at least one mandated SOTA GGUF model in
  `MODEL_SPECS` and records actual model/cache/provenance fields.
- Repair cannot be promoted unless the hard-set manifest is valid, the SOTA
  cache gate passes, and deltas improve without verifier false-accept
  regression.
- Solver cannot be promoted unless `.280` gains reproduce with larger fixed
  coverage and durable provenance.
- FR-11 cannot be promoted unless independent held-out metrics remain distinct
  from update-selection utility and negative controls/forgetting guards pass.
- GateMate/SSQA cannot be promoted beyond smoke/projection without host-visible
  readback or sample/timing evidence.
- Matrix and capstone artifacts use terminal `honest_verdict` prefixes and mark
  blocked/flagged rows honestly.

## Failed-Experiment Rerun Compliance

The YAML tasks include `prior_failures` where `.281` intentionally revisits a
failed, blocked, or flagged scope:

- `exp2977` for SOTA repair cache blockage.
- `exp2980` for solver gains with insufficient provenance.
- `exp2934` for AquaForte/BEAVER substrate relabeling.
- `exp2983` for trace-to-skill pilot follow-up.
- `exp2984` and `exp2985` for hardware readback/SSQA projection follow-up.
- `exp2986` and `exp2987` for matrix/capstone synthesis.

Each entry includes `retire_if_same_verdict: true`, so repeated failure
mechanically retires the scope instead of recycling it.

## Out of Scope

- External paper submission, arXiv upload, institutional publication, or
  operator-facing release action.
- Any claim that Carnot has demonstrated FPGA sampler acceleration,
  thermodynamic hardware behavior, or KV260/GateMate speedup from the planned
  smoke/readback tasks alone.
- Replacing exact verifiers with LLM judges.
- Treating telemetry, attribution, first-token confidence, or semantic-energy
  proxies as final correctness evidence.
- Downloading or switching to non-mandated headline models when the required
  local GGUF cache is unavailable.
