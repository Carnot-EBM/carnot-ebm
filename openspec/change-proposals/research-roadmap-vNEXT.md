# Research Roadmap vNEXT: Milestone 2026.05.278

**Title:** Structured Code Repair + Utility-Gated Self-Learning + GateMate Materialization

**Planned:** 2026-05-23

**Previous milestone:** 2026.05.277

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.277 Proved

Milestone `.277` completed the Deep Think corrigenda and made the Phase-3 /
paper-v6 boundary sharper rather than broader. The authoritative terminal
artifact is `results/experiment_2948_capstone_v277.json`.

- `paper_ready=true`, but the headline is explicitly narrow.
- KV260 samples are distinguishable from CPU sequential Gibbs; paper-v6 must not
  claim exact Boltzmann sampling on the current KV260 path.
- The same-schedule CPU comparator recorded `kv260_same_schedule_speedup=0.98225`
  at n=64, so no current KV260 speedup claim survives on that basis.
- The code-corpus verifier result is useful: AUPRC remained strong at 0.888889
  against a low-base-rate negative corpus.
- The SOTA code-generation continuation ran with 50 tasks but remained weak
  (`pass@1=0.0600`, `pass@k=0.1600`), making generation/repair the main
  product gap rather than verifier scoring.
- FR-11 gained a nonuniform continuation replay curriculum pilot, but not yet a
  held-out utility gate or rollback rule.
- PolarFire reached a 500-clause constraint-scorer hash-verification artifact.
- GateMate still has no Carnot Ising tile bitstream/flash artifact; the known
  blocker is constraints/bitstream materialization, not the old obsolete
  `nextpnr-gatemate` tool assumption.

## Three Biggest Gaps

### Gap 1: Verifier Signal Has Not Improved Generation

Carnot can rank or reject code candidates, but `.277` showed that local SOTA
generation is still poor. The next milestone should turn the code hallucination
taxonomy and AUPRC signal into repair prompts, constrained candidate manifests,
and a measured pass@1/pass@k delta under the mandated local GGUF models.

### Gap 2: FR-11 Is Still a Scheduler, Not a Proven Self-Learner

The replay curriculum pilot is useful, but it does not yet prove continuous
self-learning as required by the PRD. `.278` needs a utility-gated replay loop
with held-out improvement, forgetting checks, and rollback semantics before any
"learns from verification" claim can be upgraded.

### Gap 3: Hardware Evidence Is Narrow and Incomplete

KV260 is now bounded as a functional fixed-schedule heuristic path, not a
Boltzmann/speedup result. GateMate remains the most actionable missing hardware
artifact, while PolarFire has a clean 500-clause score but not a larger
continuation. `.278` should materialize GateMate constraints/bitstream/flash
evidence and extend PolarFire carefully, with hash/timing/smoke claims only.

## New Research Integrated

The 2026-05-23 post-`.277` sweep appended these items to
`research-references.md`:

- **TruncProof** (arXiv:2605.13076): LL(1)-based JSON completion under token
  budgets; informs bounded repair manifests and certificate emission.
- **`guidance-ai/llguidance`**: practical local-first JSON/CFG/regex
  constrained decoding for llama.cpp, vLLM, SGLang, and related runtimes;
  informs structured candidate manifests for local GGUF runs.
- **MAZE adaptive constrained code generation**: supports the shift from
  "generate then verify" to "compile constraints before decoding" for code.
- **Energy-Guided Decoding and Spilled Energy**: useful training-free energy
  telemetry for candidate triage, but not headline evidence given prior Carnot
  energy-telemetry limitations.
- **Soft-Radial Projection** (arXiv:2602.03461): informs future differentiable
  constraint-preserving self-learning, included in `.278` as a design note for
  utility-gated replay rather than a full new neural layer.
- **Lagrange oscillatory neural networks** (2025 / arXiv:2505.07179): reinforces
  feasibility-first energy accounting for hardware constraint systems.
- **Extropic/THRML and Logical Intelligence/Kona public updates**: remain useful
  architecture context only; no local TSU/Kona access or reproducible benchmark
  changed Carnot's evidence base.

## Architecture Snapshot

```text
        +---------------------------------------------------------+
        | Phase A: structured code repair and operating point     |
        |                                                         |
        | exp2949 archive/activate                               |
        | exp2950 taxonomy repair prompt manifest                |
        | exp2951 structured candidate manifest adapter           |
        | exp2952 gated SOTA repair evaluation                    |
        | exp2953 code verifier threshold policy                  |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase B: self-learning and exact verification            |
        |                                                         |
        | exp2954 FR-11 utility-gated replay curriculum           |
        | exp2959 NL-to-Z3 execution repair mini                  |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase C: hardware materialization                       |
        |                                                         |
        | exp2955 GateMate constraints materialization            |
        | exp2956 gated GateMate n=16 bitstream build             |
        | exp2957 gated GateMate flash/timing smoke               |
        | exp2958 PolarFire 1000-clause scorer continuation       |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase D: matrix and closeout                            |
        |                                                         |
        | exp2960 cross-corpus matrix v12                         |
        | exp2961 capstone .278                                   |
        +---------------------------------------------------------+
```

## Phase Structure

### Phase A: Structured Code Repair and Operating Point

- `exp2949` archives `.277` and activates `.278`.
- `exp2950` converts `.277` code failures, taxonomy rows, and AUPRC results into
  a repair-prompt manifest for the mandated local GGUF models. It is a planning
  artifact, not a pass-rate claim.
- `exp2951` builds a structured candidate manifest adapter around JSON schema /
  CFG constraints, preferring `llguidance` or local grammar support when present
  and falling back to deterministic schema validation when not.
- `exp2952` runs the actual SOTA repair evaluation only if the repair manifest
  and structured manifest adapter are ready. It measures pass@1/pass@k deltas,
  syntax failures, schema failures, and verifier acceptance on the same bounded
  code row that `.277` exposed as weak.
- `exp2953` converts the strong code AUPRC row into a threshold policy with
  PPV/recall/cost tradeoffs and explicit deployment boundaries.

### Phase B: Self-Learning and Exact Verification

- `exp2954` is the required continuous self-learning task. It adds a
  utility-gated replay update, held-out improvement metric, forgetting guard,
  and rollback rule over the `.277` replay-curriculum lineage.
- `exp2959` repairs the blocked LLMEval-Logic/Z3 mini path by making Z3
  execution, parseability, solver authority, and local GGUF proposal provenance
  explicit. It must not treat LLM answers as ground truth.

### Phase C: Hardware Materialization

- `exp2955` materializes the minimal GateMate n=16 constraints and test vectors
  that were missing in `.276`, using the corrected himbaechel/gmpack toolchain.
- `exp2956` builds a GateMate n=16 bitstream only if `exp2955` proves the
  constraints and toolchain are ready.
- `exp2957` performs a bounded flash/timing smoke only if `exp2956` produces a
  bitstream. It may report board contact, flash transcript, timing, and output
  hash; no speedup or thermodynamic sampling claim is allowed.
- `exp2958` extends the PolarFire scorer from 500 to 1000 clauses with hash and
  transcript evidence. It is a scoring/hash continuation, not a broad hardware
  acceleration claim.

### Phase D: Matrix and Closeout

- `exp2960` rebuilds cross-corpus matrix v12 with `.278` code-repair,
  self-learning, solver, hardware, and claim-boundary rows.
- `exp2961` synthesizes `.278`, classifies every artifact, records which gaps
  closed, and recommends the next milestone direction.

## Dependency Graph

```text
exp2949

exp2950
  -> exp2952

exp2951
  -> exp2952

exp2952
exp2953
exp2954
exp2958
exp2959
  -> exp2960

exp2955
  -> exp2956
       -> exp2957

all artifacts
  -> exp2961
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2952` gates on:
  - `exp2950.repair_prompt_manifest_ready == true`
  - `exp2951.structured_decode_manifest_ready == true`
- `exp2956` gates on `exp2955.gatemate_constraints_ready == true`.
- `exp2957` gates on `exp2956.gatemate_bitstream_built == true`.
- `exp2960` gates on `exp2954.self_learning_utility_artifact_ready == true`.
- `exp2961` is intentionally ungated so the milestone can close honestly even
  if a branch is blocked.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` or an equivalent local GGUF runtime with GPU offload support.
- Mandated headline GGUFs available through the `cached_sota_pair()` pattern:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Legacy Qwen3.5-0.8B or gemma-4-E4B-it models may appear only as CPU smoke
  tests and cannot be headline-result models.

Required for hardware tasks:

- GateMate A1-EVB-2M attached through DirtyJTAG.
- OSS CAD Suite path containing `yosys`, `nextpnr-himbaechel --device CCGM1A1`,
  `gmpack`, and `openFPGALoader`.
- PolarFire board reachable over its documented SSH path with the `.277`
  500-clause artifact available as the baseline.

Out of scope:

- KV260 speedup, Boltzmann, or thermalization claims.
- Host-side KV260 SD-card checks.
- Extropic TSU/Z1 hardware claims.
- Kona performance claims.
