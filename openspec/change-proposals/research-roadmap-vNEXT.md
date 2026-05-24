# Research Roadmap vNEXT: Milestone 2026.05.283

**Title:** Claim Repair v2 + Feasibility-Gated Self-Learning + GateMate IO Boundary

**Planned:** 2026-05-24

**Previous milestone:** 2026.05.282

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.282 Proved

Milestone `.282` completed through
`results/experiment_3011_capstone_v282.json`. The capstone verdict is
`complete: capstone_ready=true; paper_ready=false; repaired=4; flagged=23;
blocked=9; gated_skipped=1; missing=1; next=2026.05.283 claim-repair-v2`.

- **Local SOTA GGUF execution is available but still narrow.** `exp3001`
  refreshed cache/provenance and at least one mandated headline model ran
  live. `.283` should reuse the proven path and add top-k/logprob telemetry
  only if the local loader exposes it honestly.
- **Repair has a real delta but remains non-promotable.** `exp3003` used
  `unsloth/gemma-4-26B-A4B-it-GGUF` and produced
  `pass_at_1_delta=0.4167`, but `repair_rerun_clean=false` because
  syntax/schema failure deltas were both `+0.5`. The next repair task must
  reduce those failures without increasing false accepts.
- **Metamorphic oracle construction was useful but flagged as an audit gate.**
  `exp3002` produced 59 variants over 24 source items plus false-accept and
  tautology probes. It should be treated as methodology infrastructure, not as
  a headline result.
- **AquaForte/BEAVER substrate provenance was repaired.** `exp3004` cleanly
  separated live retry from enumerator fallback and supplied durable model,
  transcript, and duration evidence. Carry it forward only as a provenance
  repair, not as a claim that the live retry solved the task.
- **Validator trees and fixed-point diagnostics are the cleanest growth path.**
  `exp3005` exact-checked 20 validator trees and `exp3006` completed a bounded
  fixed-point diagnostic with no native EqR claim.
- **FR-11 is still flagged.** `exp3007` reported
  `trace_memory_stability_ready=true`, but matrix/capstone kept it
  non-promotable because the held-out evidence is too small and still risks
  tautological scoring. `.283` needs independent feasible/infeasible separation
  and negative controls.
- **GateMate remains blocked before SSQA.** `exp3008` detected the board and
  attempted flash, but `host_visible_io_ready=false`: spin output and done
  signals remain internal RTL wires with no physical pin, UART, GPIO, JTAG,
  status register, CSR, AXI, or logic-analyzer transport. `exp3009` then
  gate-skipped and no artifact was written.
- **The capstone itself was adversarial-flagged by aggregation false positives.**
  `exp3011` did no new LLM call, but the artifact referenced upstream GGUF/CUDA
  markers and was flagged `DURATION_TOO_SHORT`. `.283` aggregation artifacts
  must keep source model metadata under cited-upstream provenance fields rather
  than top-level live-inference fields.

## Three Biggest Gaps

### Gap 1: Repair Promotion Needs an Acceptance Controller

The PRD requires verifiable reasoning, not just a positive repair delta.
`exp3003` showed the useful signal and the failure mode at the same time:
repairs improved pass rate but also raised syntax/schema failures. Cactus-style
constrained acceptance suggests a bounded accept/reject layer over candidate
repairs: accept more only when divergence and false-accept risk are controlled.

### Gap 2: FR-11 Needs Independent Feasibility Evidence

The continuous self-learning loop is still too close to grading itself. DVI
shows how verifier accept/reject events can become online supervision, while
Differentiable Symbolic Planning suggests an interpretable feasibility channel.
Carnot should test a small verifier-feedback learner over exact validator-tree
traces, with independent feasible/infeasible separation, forgetting guards, and
negative controls.

### Gap 3: GateMate Needs an Output Contract Before Any Sampler Work

GateMate cannot support SSQA, sampler, or acceleration claims until the host can
observe at least one deterministic status bit or byte. `.283` should first add
or diagnose a bounded RTL/CCF transport shim, then attempt board smoke only if
the transport exists. SSQA must always write an explicit artifact, even when the
hardware gate remains closed.

## New Research Integrated

The 2026-05-24 post-`.282` sweep appended these sources to
`research-references.md` before this design was written:

- **Cactus (arXiv:2604.04987):** constrained-acceptance speculative sampling.
  Used in `.283` as a repair-candidate acceptance controller, not a decoding
  speed claim.
- **Draft, Verify, and Improve (arXiv:2510.05421):** verifier feedback becomes
  online supervision. Used for the FR-11 verifier-feedback controller over
  cached exact traces.
- **Differentiable Symbolic Planning (arXiv:2604.02350):** learned feasibility
  channel with exact-zero rule selection. Used to de-tautologize FR-11 memory.
- **NSVIF (arXiv:2601.17789):** instruction-following verification as
  constraint satisfaction. Used to expand validator trees beyond code/solver
  rows without LLM-as-judge labels.
- **CAIM (arXiv:2602.05595):** adaptive Ising control as hardware context.
  Informative only; `.283` remains digital GateMate IO first.
- **HalluGuard (arXiv:2601.18753 via Hugging Face):** useful taxonomy split
  between data-driven and reasoning-driven hallucinations. No NTK claim unless
  the implementation actually computes the NTK/geometric substrate.
- **BEAVER (arXiv:2512.05439 via Hugging Face):** deterministic prefix-closed
  bounds. Used for a validator-tree frontier certificate with strict separation
  from live LLM evidence.
- **EBT/ARM-EBM citation watch:** supports the long-term Phase-3 direction but
  does not justify native EBT training in `.283`.
- **Extropic and Logical Intelligence updates:** strategic context only. No
  local TSU/Z1/Kona/Aleph performance claim is allowed.

## Architecture Snapshot

```text
                  exp3011 capstone v282 / matrix v16
                                  |
                                  v
                    exp3012 archive .282 and activate .283
                                  |
                                  v
              exp3013 SOTA GGUF + logprob/top-k preflight
                         |                         \
                         |                          \
                         v                           v
      exp3014 repair failure taxonomy      exp3017 NSVIF instruction
                         |                  validator-tree expansion
                         v                           |
      exp3015 Cactus acceptance controller           v
                         |                  exp3018 BEAVER frontier
                         v                  certificate
      exp3016 SOTA repair rerun                       |
                         |                            v
                         |                  exp3019 feasibility-channel
                         |                  FR-11 diagnostic
                         |                            |
                         |                            v
                         |                  exp3020 DVI verifier-feedback
                         |                  self-learning controller
                         |                            |
                         +-------------> exp3024 matrix v17 <-------------+
                                                                            |
   Hardware branch:                                                          |
      exp3021 GateMate RTL/CCF host-visible transport shim                   |
              |                                                             |
              v                                                             |
      exp3022 GateMate transport flash/smoke v3                              |
              |                                                             |
              v                                                             |
      exp3023 SSQA explicit gate artifact / RTL report ----------------------+
                                                                            |
                                  exp3025 capstone v283
```

## Phase Structure

### Phase A: Activation and Live-Model Boundary

1. **exp3012 - Archive `.282` and activate `.283`.**
   Archive `.282`, preserve flagged/blocked/missing rows, and activate the new
   queue without modifying `scripts/research_conductor.py`.

2. **exp3013 - SOTA GGUF logprob/top-k telemetry preflight.**
   Reuse the mandated local SOTA cache and determine whether the loader exposes
   enough top-k/logprob telemetry for Cactus-style candidate acceptance. Legacy
   small models remain smoke-only.

3. **exp3014 - Repair syntax/schema failure taxonomy.**
   Diagnose why `exp3003` improved pass rate while increasing syntax/schema
   failures. Use HalluGuard's data-vs-reasoning split as taxonomy only.

### Phase B: Repair Acceptance and Formal Validator Expansion

4. **exp3015 - Cactus-style repair acceptance controller.**
   Build an offline accept/reject controller over cached candidates and
   metamorphic variants. It must reduce syntax/schema promotion risk without
   increasing false accepts.

5. **exp3016 - Gated SOTA repair rerun with acceptance controller.**
   Run live repair only if the SOTA telemetry and acceptance-controller gates
   pass. Promotion requires positive deltas, clean false accepts, and no
   syntax/schema regression.

6. **exp3017 - NSVIF-style instruction validator tree expansion.**
   Extend the exact-check validator tree corpus into small instruction-following
   constraints with logical/runtime authority where possible.

7. **exp3018 - BEAVER-style validator frontier certificate.**
   Produce deterministic frontier/probability-bound style evidence over the
   validator-tree corpus. Keep live LLM retry and enumerator fallback separate.

### Phase C: Continuous Self-Learning and Hardware Boundary

8. **exp3019 - Feasibility-channel FR-11 de-tautology diagnostic.**
   Add a DSP-inspired feasibility channel over exact traces and test whether it
   separates feasible from infeasible cases on independent metrics.

9. **exp3020 - DVI verifier-feedback self-learning controller.**
   Continuous self-learning task. Convert verifier accept/reject events into a
   small online controller and test held-out utility, forgetting, drift, and
   negative controls.

10. **exp3021 - GateMate RTL/CCF host-visible transport shim.**
    Add or precisely diagnose physical output binding for `spin_out`/`done`.
    No board sampler, thermalization, or speedup claim.

11. **exp3022 - GateMate transport flash/smoke v3.**
    Attempt board flash/smoke only if the transport shim is ready, then capture
    deterministic host-visible bytes or a precise blocked transcript.

12. **exp3023 - SSQA explicit gate artifact and RTL report.**
    Always write an SSQA artifact. If the GateMate gate is still closed, record
    explicit gate-skipped status; if it is open, produce bounded RTL/PnR/resource
    evidence with no sampler or speedup claim.

### Phase D: Synthesis and Go/No-Go

13. **exp3024 - Cross-corpus matrix v17.**
    Aggregate `.283` honestly, classify every row, and avoid top-level
    live-inference metadata in aggregation-only artifacts.

14. **exp3025 - Milestone `.283` capstone.**
    Decide whether repair, FR-11, and GateMate/SSQA are promotable. Keep
    `paper_ready=false` unless every promotion gate is clean.

## Dependency Graph

```text
exp3012
  -> exp3013
       -> exp3016
  -> exp3014 -> exp3015 -> exp3016
  -> exp3017 -> exp3018 -> exp3019 -> exp3020
  -> exp3021 -> exp3022 -> exp3023
exp3016 -> exp3024
exp3018 -> exp3024
exp3020 -> exp3024
exp3023 -> exp3024
exp3024 -> exp3025
```

Structured conductor gates are included for:

- `exp3015`: requires `exp3014.repair_failure_taxonomy_ready == true`.
- `exp3016`: requires `exp3013.sota_logprob_ready == true` and
  `exp3015.acceptance_controller_ready == true`.
- `exp3018`: requires `exp3017.instruction_validator_tree_ready == true`.
- `exp3019`: requires `exp3018.frontier_certificate_ready == true`.
- `exp3020`: requires `exp3019.feasibility_channel_diagnostic_ready == true`.
- `exp3022`: requires `exp3021.gatemate_transport_rtl_ready == true`.
- `exp3025`: requires `exp3024.matrix_v17_ready == true`.

`exp3023` intentionally has no structured gate. It always writes an artifact so
the `.282` missing-SSQA pattern does not recur.

## Hardware Requirements

- **Dual RTX 3090 CUDA host:** required for `exp3013` and `exp3016` live local
  SOTA GGUF work. At least one of `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF` must
  produce a live transcript for headline fields.
- **CPU-only path:** acceptable for archive, repair taxonomy, offline
  acceptance controller, validator expansion, frontier certificate,
  feasibility diagnostics, FR-11 controller over cached traces, matrix, and
  capstone.
- **GateMate A1:** required for `exp3022`. `exp3021` may be RTL/CCF/toolchain
  only; `exp3022` is the first board-facing task and must record board
  detection, flash status, and host-visible bytes/status or a precise blocker.
- **SSQA/GateMate RTL/PnR:** `exp3023` may produce RTL/PnR/resource evidence
  only within the output-contract boundary. It must not claim speedup,
  thermalization, Boltzmann sampling, or FPGA acceleration.
- **Extropic/THRML, CAIM, Kona/Aleph:** architecture context only. No `.283`
  task depends on authenticated TSU/Z1/XTR-0, analog Ising hardware, or Kona
  internals.

## Acceptance Criteria

- `research-references.md` contains the post-`.282` sweep before the roadmap
  design.
- `research-roadmap-next.yaml` declares milestone `2026.05.283` and leaves
  `research-roadmap.yaml` unchanged.
- Every live LLM task includes the mandated SOTA GGUF models in `MODEL_SPECS`
  and records model/cache/provenance fields. Legacy models remain smoke-only.
- Repair cannot be promoted unless pass-rate deltas are positive,
  false-accept deltas are non-positive, tautology probes are clean, and
  syntax/schema failures do not regress.
- FR-11 cannot be promoted unless the verifier-feedback controller improves
  independent held-out metrics, rejects negative controls, preserves forgetting
  guards, and records a non-tautological feasibility channel.
- GateMate cannot be promoted unless host-visible output exists or the blocker
  is precisely diagnosed. SSQA must emit an explicit artifact even when gated.
- Matrix and capstone aggregation artifacts must use
  `inference_substrate=aggregation_from_upstream_artifacts` and avoid top-level
  live model metadata that would trigger false `DURATION_TOO_SHORT` flags.

## Failed-Experiment Rerun Compliance

Carry-forward tasks include `prior_failures` entries with mandatory
`retire_if_same_verdict: true` for the relevant blocked/flagged lineages:

- `exp3003` flagged repair methodology and syntax/schema regression.
- `exp3007` flagged FR-11 trace-memory stability.
- `exp3008` blocked GateMate host-visible IO.
- `exp3009` gated-skipped/missing SSQA artifact.
- `exp3010` / `exp3011` synthesis rows that ended with `paper_ready=false` and
  aggregation false positives.

No task depends on a retired upstream ID from `ops/exclusion_manifest.yaml`.

## Out of Scope

- External publication, arXiv submission, Hugging Face public release action, or
  public announcement.
- New WOPR/game cartridges, GRPO/VPRM, HardNet++/DSP, SpecAnn, PIMI, OTV, or
  KV260 host-SD-card scopes.
- Claiming Extropic/TSU/Z1/XTR-0, analog Ising hardware, Kona/Aleph, photonic,
  or quantum hardware access.
- Claiming GateMate/KV260 acceleration, thermalization, or Boltzmann sampling
  without board-visible sample/timing evidence.
- Treating LLM judges, HalluGuard name reuse, NTK claims, metamorphic
  consistency, or prompt schemas as substitutes for executable verifiers.
