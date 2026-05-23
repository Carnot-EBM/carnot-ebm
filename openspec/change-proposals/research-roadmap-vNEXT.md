# Research Roadmap vNEXT: Milestone 2026.05.275

**Title:** Hardware Baselines + Code Hallucination Repair + Verifier-Grounded Self-Learning

**Planned:** 2026-05-23

**Previous milestone:** 2026.05.274

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.274 Proved

Milestone `.274` reactivated the physical hardware portfolio while keeping
claim boundaries explicit. The authoritative terminal artifact is
`results/experiment_2908_capstone_v274.json`.

- `paper_ready=true`, with 8 clean artifacts, 1 flagged artifact, 1 blocked
  artifact, 0 missing artifacts, and 1 pilot-only artifact.
- KV260 produced the first real board-level n=64 Ising sampler latency
  transcript. The clean row is `exp2898`: overlay `carnot_ising_v2_n64`,
  `/dev/uio0-4`, three seeds, 10,000-sample p50 latency around 24 us, and a
  bitstream hash. No speedup claim is allowed yet because there is no same-basis
  CPU Gibbs baseline.
- GateMate A1-EVB-2M remained blocked. `exp2899` honestly reported
  `blocked_gatemate_toolchain_missing` because `nextpnr-gatemate` was absent.
  No bitstream exists, and no flash attempt is allowed until build provenance is
  clean.
- PolarFire SoC produced a clean RISC-V dispatch smoke with hash-verified
  constraint scoring. It is CPU-side dispatch evidence only, not fabric
  acceleration.
- THRML import was repaired locally and n=16 parity passed without making a
  hardware claim.
- Cross-corpus matrix v8 aggregated forward-only evidence and kept blocked,
  flagged, and pilot-only rows separate.
- The SOTA code-generation expansion is flagged. `exp2905` used the mandated
  local GGUF path but reported `pass@1=0.5000` and `pass@k=0.5000` on only two
  tasks per corpus, omitted a top-level `random_seed`, and is not headline
  eligible.
- FR-11 hardware replay is pilot-only. `exp2906` validated a KV260 replay
  dispatch path but did not prove a repeated self-learning gain or hardware
  replay speedup.

## Three Biggest Gaps

### Gap 1: Hardware Acceleration Still Lacks Matched Baselines

KV260 now has real latency, but Carnot cannot say anything about speedup until a
same-basis CPU Gibbs baseline uses the exact n=64 coupling/field tensors, sparse
topology, seeds, and sample counts. GateMate is one step earlier: it still needs
toolchain provenance and a built bitstream before any physical-board step.

### Gap 2: Generated-Code Evidence Is Methodologically Weak

The `exp2905` code row is flagged because the sample size is too small, pass@1
and pass@k are tautologically equal, and the artifact lacks a top-level random
seed. Recent code-hallucination work suggests Carnot should stop treating code
generation as only pass/fail and instead classify invented imports, undefined
names, invented methods, and invalid arguments under deterministic verifiers.

### Gap 3: FR-11 Has Dispatch and Memory Pieces, Not Continuous Learning Proof

FR-11 has RecMem-style memory, fast/slow replay evidence, and a KV260 dispatch
pilot. It still lacks an end-to-end verifier-grounded online update that uses
verified trajectories as dense process rewards, measures forgetting, and reports
whether energy or correctness improves after replay.

## New Research Integrated

The 2026-05-23 post-`.274` sweep appended these items to
`research-references.md`:

- **Spilled Energy in Large Language Models** (arXiv:2602.18671 / ICLR 2026):
  training-free logit energy for hallucination localization; use as a small
  local-GGUF detector micro-panel only when logprob provenance is clean.
- **Delulu** (arXiv:2605.07024): verified multi-language code hallucination
  taxonomy and container-checked FIM samples; use its four hallucination
  classes to repair `exp2905` methodology.
- **RWOPD for NL-to-SVA** (arXiv:2605.13501): verifier-weighted on-policy
  distillation with SymbiYosys+Z3 property-equivalence checks; use as the
  FR-11 pattern for weighting self-learning by verified behavior.
- **Evidence Over Plans** (arXiv:2605.09192) and **Verifiable Process Rewards**
  (arXiv:2605.10325): verified trajectories should produce dense process
  signals, not only terminal pass/fail memories.
- **ConstraintBench** (arXiv:2602.22465): feasibility and optimality must be
  separated for direct constrained optimization; use a small local mini-benchmark.
- **OpenComputer** (arXiv:2605.19769): app-specific state verifiers outperform
  LLM judges when task success depends on structured state; implement a local
  state-verifier harness.
- **llguidance**: local constrained decoding for JSON schema, regex, and CFGs;
  use only as optional structured-output support for mandated GGUF models.
- **Extropic TSU / THRML status**: continue simulator parity and portability,
  but make no TSU hardware claim.

## Architecture Snapshot

```text
        +-------------------------------------------------------+
        | Phase A: evidence repair and code verification        |
        |                                                       |
        | exp2909 archive/activate                             |
        | exp2910 SOTA codegen corrigendum v2                  |
        | exp2911 Delulu-style code hallucination verifier     |
        +-------------------------+-----------------------------+
                                  |
                                  v
        +-------------------------------------------------------+
        | Phase B: hardware baselines and sampler portability   |
        |                                                       |
        | exp2912 KV260 same-basis CPU Gibbs baseline          |
        | exp2913 KV260 hardware-vs-CPU claim boundary         |
        | exp2914 GateMate toolchain preflight                 |
        | exp2915 GateMate n=16 bitstream build                |
        | exp2916 THRML-KV260 sampler parity                   |
        +-------------------------+-----------------------------+
                                  |
                                  v
        +-------------------------------------------------------+
        | Phase C: verifier-grounded learning and benchmarks    |
        |                                                       |
        | exp2917 spilled-energy logit detector micro-panel    |
        | exp2918 FR-11 verifiable process rewards             |
        | exp2919 ConstraintBench mini direct-optimization row  |
        | exp2920 OpenComputer-style state-verifier harness    |
        +-------------------------+-----------------------------+
                                  |
                                  v
        +-------------------------------------------------------+
        | Phase D: matrix, paper boundary, and closeout         |
        |                                                       |
        | exp2921 cross-corpus matrix v9 + paper boundary      |
        | exp2922 capstone .275                                |
        +-------------------------------------------------------+
```

## Phase Structure

### Phase A: Evidence Repair and Code Verification

- `exp2909` archives `.274` and activates `.275`.
- `exp2910` reruns the bounded SOTA code-generation row with the mandated local
  GGUF models, a top-level random seed, `n_tasks_per_corpus >= 20`, and
  methodology fields that explicitly address the pass@1/pass@k tautology.
- `exp2911` consumes the corrected code-generation artifact and adds a
  Delulu-style hallucination taxonomy plus static/runtime verifiers for invented
  imports, undefined names, invented attributes/methods, and invalid arguments.

### Phase B: Hardware Baselines and Sampler Portability

- `exp2912` creates the same-basis CPU Gibbs baseline for the KV260 n=64 Ising
  problem using the exact seeds and tensors from `exp2898`.
- `exp2913` compares KV260 and CPU only after `exp2912` is ready. It may compute
  speedup eligibility, but it must write a claim-boundary field even if no
  speedup is defensible.
- `exp2914` checks GateMate toolchain readiness and records how `nextpnr-gatemate`
  is found or why it remains blocked.
- `exp2915` builds the n=16 GateMate Ising tile only if `exp2914` reports the
  toolchain ready. It records synthesis, place-and-route, and bitstream hash;
  flashing remains out of scope.
- `exp2916` compares THRML simulation, CPU Gibbs, and KV260 evidence on the
  same problem basis after the CPU baseline is present. It remains a simulator
  parity task, not a TSU hardware claim.

### Phase C: Verifier-Grounded Learning and Benchmarks

- `exp2917` runs a small spilled-energy logit detector micro-panel on mandated
  local GGUF outputs and reports whether energy signals separate verified from
  hallucination-like outputs. It cannot claim a benchmark.
- `exp2918` is the mandatory continuous self-learning experiment. It converts
  verified code and hardware trajectories into dense process rewards and tests a
  bounded FR-11 online replay update with forgetting checks.
- `exp2919` materializes a ConstraintBench-style mini benchmark for direct
  constrained optimization under local GGUF models, reporting feasibility and
  optimality separately.
- `exp2920` creates a lightweight OpenComputer-style state-verifier harness for
  local agentic tasks, using structured state checks and auditable partial
  credit rather than LLM-as-judge.

### Phase D: Matrix, Paper Boundary, and Closeout

- `exp2921` rebuilds cross-corpus matrix v9 and stages the paper-v6 claim
  boundary. It consumes clean rows only and carries blocked/flagged/pilot rows
  forward explicitly.
- `exp2922` synthesizes `.275`, classifies every artifact, records paper and
  hardware claim eligibility, and recommends the next milestone direction.

## Dependency Graph

```text
exp2909
  -> exp2910
       -> exp2911

exp2912
  -> exp2913
  -> exp2916

exp2914
  -> exp2915

exp2911 + exp2912
  -> exp2918

exp2917
exp2919
exp2920

exp2911 + exp2913 + exp2918 + exp2919 + exp2920
  -> exp2921

all clean/flagged/blocked/pilot artifacts
  -> exp2922
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2911` gates on `exp2910.codegen_corrigendum_ready == true`.
- `exp2913` gates on `exp2912.same_basis_cpu_baseline_ready == true`.
- `exp2915` gates on `exp2914.gatemate_toolchain_ready == true`.
- `exp2916` gates on `exp2912.same_basis_cpu_baseline_ready == true`.
- `exp2918` gates on:
  - `exp2911.code_hallucination_verifier_ready == true`
  - `exp2912.same_basis_cpu_baseline_ready == true`
- `exp2921` gates on:
  - `exp2911.code_hallucination_verifier_ready == true`
  - `exp2913.kv260_claim_boundary_ready == true`
  - `exp2918.online_self_learning_ready == true`
  - `exp2919.constraintbench_mini_ready == true`
  - `exp2920.state_verifier_harness_ready == true`
- `exp2922` is intentionally ungated so the milestone can close honestly even
  when a branch is blocked.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` with GPU offload support.
- Mandated headline GGUFs available through `cached_sota_pair()`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`

Required for hardware tasks:

- KV260 reachable through `ssh kria`, with the `.274` overlay evidence from
  `results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json`.
- GateMate A1-EVB-2M attached through DirtyJTAG, but `exp2914` must prove
  `nextpnr-gatemate` before `exp2915` runs.
- THRML import repaired locally per
  `results/experiment_2901_thrml_local_import_repair_v1.json`.

Out of scope:

- Host-side KV260 SD-card manipulation.
- GateMate flashing before a built bitstream hash exists.
- Extropic TSU/Z1/XTR-0 hardware claims.
- AMD XDNA/NPU tasks unless a later directive reopens them.

## Agent Routing

- Codex/gpt-5.5 is assigned to formulaic code, verifier, sampler, and hardware
  harness tasks.
- Claude Opus is reserved for the capstone synthesis because `.274` already
  needed a rescue on capstone-style work.
- Live LLM experiments must use the mandated SOTA local GGUFs in `MODEL_SPECS`.
  Legacy tiny models are allowed only as CPU smoke-tests and cannot produce
  headline rows.

## Acceptance Criteria

The milestone is successful if:

- `.274` is archived and `.275` activates cleanly.
- `exp2910` either produces a clean corrected code-generation artifact or writes
  an honest blocked artifact with no headline claim.
- `exp2912` produces a same-basis CPU baseline for KV260, and `exp2913` records
  a hardware claim boundary.
- GateMate either builds an n=16 bitstream with hash/provenance or remains
  honestly blocked at the toolchain preflight gate.
- At least one continuous self-learning artifact (`exp2918`) reports
  verifier-grounded replay metrics and forgetting checks.
- Matrix v9 and the `.275` capstone preserve clean/flagged/blocked/pilot-only
  distinctions without upgrading a row by implication.
