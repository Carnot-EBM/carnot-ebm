# Research Roadmap vNEXT: Milestone 2026.05.276

**Title:** Evidence Boundary Repair + GateMate Bring-Up + Solver-Grounded Self-Learning

**Planned:** 2026-05-23

**Previous milestone:** 2026.05.275

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.275 Proved

Milestone `.275` closed the hardware-baseline and verifier-grounded
self-learning gap opened by `.274`. The authoritative terminal artifact is
`results/experiment_2922_capstone_v275.json`.

- `paper_ready=true` and `hardware_speedup_claim_eligible=true`.
- KV260 now has a defensible hardware-vs-CPU claim boundary. The same-basis
  CPU Gibbs baseline and KV260 comparison produced an n=64 sparse Ising speedup
  claim around 18.65-19.24x by matched sample count, with board transcript and
  `/dev/uio` evidence.
- The SOTA code-generation row was repaired enough to produce a clean local
  GGUF row (`exp2910`) using the mandated model path, 40 tasks, and 320 captured
  candidates.
- FR-11 gained a clean verifier-grounded replay scheduler row (`exp2918`):
  positive replay delta, energy-proxy improvement, no measured forgetting, and
  explicit "scheduler/replay only" boundaries.
- The OpenComputer-style state-verifier harness (`exp2920`) produced a clean
  local structured-state verification artifact.
- GateMate remained blocked because the preflight looked for the obsolete
  `nextpnr-gatemate` entrypoint. It nevertheless discovered the actionable
  current toolchain path: `nextpnr-himbaechel --device CCGM1A1` plus `gmpack`.
- Three artifacts are useful but not yet claim-clean:
  - `exp2911` code hallucination taxonomy verifier is deterministic and useful,
    but flagged because aggregation over upstream live-model outputs looked like
    a too-fast live-model run and lacked provenance/checksum fields.
  - `exp2919` ConstraintBench mini used a mandated local GGUF, but feasibility
    and syntax-validity matched tautologically and the run was too short for a
    live-model headline row.
  - `exp2921` matrix v9 is aggregation-only, but inherited compute-bound
    adversarial flags because it did not declare aggregation provenance clearly
    enough.

## Three Biggest Gaps

### Gap 1: Claim-Clean Evidence Boundaries Lag Behind Working Artifacts

The code taxonomy, ConstraintBench, matrix, and capstone evidence exist, but
two rows and one aggregation matrix are still adversarially flagged. `.276`
should first repair provenance, runtime, checksum, and non-tautology fields so
Carnot can distinguish real flagged science from metadata false positives.

### Gap 2: Structured Local-SOTA Generation Still Fails at Syntax and Feasibility

The strongest local code row shows syntax errors dominate candidate failures,
and ConstraintBench proved local GGUF direct optimization is feasibility-limited.
Recent BEAVER, AquaForte, LLMEval-Logic, and `llguidance` work all point to the
same direction: let the LLM propose, but constrain or verify every structured
object with exact parsers, schemas, Z3, tests, or field-level checkers.

### Gap 3: Hardware and Self-Learning Need a Second Clean Layer

KV260 has a clean n=64 claim. GateMate has tool discovery but no bitstream, and
FR-11 has replay scheduling but not structural online learning. `.276` should
turn the corrected GateMate toolchain into at least a hashed n=16 build path
and test a KAN-style continual self-learning mechanism with explicit forgetting
guards.

## New Research Integrated

The 2026-05-23 post-`.275` sweep appended these items to
`research-references.md`:

- **BEAVER** (arXiv:2512.05439): deterministic sound probability bounds for LLM
  constraint satisfaction; informs prefix-closed audit fields for structured
  generation.
- **AquaForte** (arXiv:2601.04675 / AAAI 2026): LLM-guided quantified SMT
  instantiation with solver-preserved soundness; informs local GGUF proposal +
  Z3 validation loops.
- **p-bit SSQA dual-BRAM FPGA annealing** (arXiv:2602.16143) and
  **hardware-aware SSA** (arXiv:2601.18007): inform KV260/GateMate scaling
  projections without becoming Carnot hardware evidence.
- **LLMEval-Logic** (arXiv:2605.19597 / Hugging Face Papers): forward-authored
  natural-language logic benchmark with Z3-verified formalizations and
  adversarial hardening; informs solver-checked local GGUF logic experiments.
- **NRGPT** and **LogicReward** (OpenReview ICLR 2026): support the
  energy-based reasoning and step-level verifier reward framing, but do not
  justify training a new EBT/NRGPT model locally.
- **CiteTracer and large-scale non-existent citation audits** (arXiv:2605.07723
  and arXiv:2605.08583): provide a compact field-verifiable hallucination
  benchmark for `.276`.
- **KAC** (CVPR 2025 / arXiv:2503.21076) and **COOL** (CCF Transactions 2026):
  motivate a KAN structural-memory self-learning probe with non-forgetting
  checks.
- **Extropic/THRML, Semantic Scholar EBT/ARM, and Logical Intelligence/Kona
  checks**: no new local hardware/API evidence was found; keep these as
  background context only.

## Architecture Snapshot

```text
        +---------------------------------------------------------+
        | Phase A: evidence boundary repair                      |
        |                                                         |
        | exp2923 archive/activate                               |
        | exp2924 aggregation metadata corrigendum               |
        | exp2925 code taxonomy provenance corrigendum           |
        | exp2926 ConstraintBench constrained-output rerun        |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase B: hardware bring-up and sampler scaling          |
        |                                                         |
        | exp2927 GateMate himbaechel + constraints preflight     |
        | exp2928 GateMate n=16 bitstream build                  |
        | exp2929 GateMate flash/timing boundary smoke           |
        | exp2930 KV260 p-bit/SSQA scaling projection            |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase C: solver-grounded generation and self-learning   |
        |                                                         |
        | exp2931 LLMEval-Logic Z3 mini                          |
        | exp2932 citation hallucination field verifier          |
        | exp2933 KAN continual self-learning probe              |
        | exp2934 AquaForte/BEAVER-style reformulation pipeline  |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase D: matrix and closeout                           |
        |                                                         |
        | exp2935 cross-corpus matrix v10                        |
        | exp2936 capstone .276                                  |
        +---------------------------------------------------------+
```

## Phase Structure

### Phase A: Evidence Boundary Repair

- `exp2923` archives `.275` and activates `.276`.
- `exp2924` repairs aggregation-only adversarial metadata for matrix v9 and the
  `.275` capstone. It must not rerun live inference; the deliverable is a
  provenance/checksum corrigendum over upstream artifacts.
- `exp2925` repairs the deterministic code hallucination taxonomy row by adding
  upstream model provenance, source artifact checksums, reproducibility checksum,
  and explicit `deterministic_verifier_no_new_llm_call` methodology.
- `exp2926` reruns ConstraintBench mini with constrained output, separate
  syntax/feasibility/optimality denominators, a longer live-model duration, and
  the mandated local SOTA GGUF path.

### Phase B: Hardware Bring-Up and Sampler Scaling

- `exp2927` replaces the obsolete GateMate preflight assumption with the current
  `nextpnr-himbaechel --device CCGM1A1` plus `gmpack` path and materializes or
  locates the minimal constraints needed for n=16 build attempts.
- `exp2928` builds a GateMate n=16 Ising bitstream only if `exp2927` proves the
  corrected toolchain and constraints are ready. It records synthesis, PnR,
  packing, and bitstream hashes.
- `exp2929` performs a bounded GateMate flash/timing boundary smoke only if
  `exp2928` produced a bitstream. It may report board contact and timing
  transcript, but no speedup claim.
- `exp2930` uses real KV260 n=64 evidence plus SSQA/p-bit literature to produce
  a resource and memory scaling projection for n=128/n=256. It is projection
  evidence only, not new hardware acceleration evidence.

### Phase C: Solver-Grounded Generation and Self-Learning

- `exp2931` runs a small LLMEval-Logic-style natural-language-to-formal mini
  benchmark under mandated local GGUF models and Z3 checks. It reports answer
  accuracy, parseability, Z3 execution, and rubric-like faithfulness separately.
- `exp2932` builds a citation hallucination verifier using real/fake citation
  field mutations, deterministic field matching, and local GGUF responses. It
  reports citation extraction, field-level support, and hallucination categories.
- `exp2933` is the mandatory continuous self-learning experiment. It tests a
  KAN/KAC-inspired structural memory update on a constraint stream, comparing
  replay-only scheduling against per-knot/RBF importance updates with utility
  and forgetting guards.
- `exp2934` implements a bounded AquaForte/BEAVER-style reformulation loop:
  local GGUF proposal, schema/grammar reformulation, exact verifier rejection or
  acceptance, and prefix-closed audit fields where feasible.

### Phase D: Matrix and Closeout

- `exp2935` rebuilds cross-corpus matrix v10 with clean aggregation metadata,
  separating clean, flagged, blocked, projection-only, and pilot-only rows.
- `exp2936` synthesizes `.276`, classifies every artifact, records paper and
  hardware claim eligibility, and recommends the next milestone direction.

## Dependency Graph

```text
exp2923

exp2924
  -> exp2935

exp2925
  -> exp2935

exp2926
  -> exp2934
  -> exp2935

exp2927
  -> exp2928
       -> exp2929

exp2930

exp2931
exp2932
exp2933
  -> exp2935

all clean/flagged/blocked/projection artifacts
  -> exp2936
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2928` gates on `exp2927.gatemate_himbaechel_ready == true`.
- `exp2929` gates on `exp2928.gatemate_bitstream_built == true`.
- `exp2934` gates on `exp2926.constraintbench_corrigendum_ready == true`.
- `exp2935` gates on:
  - `exp2924.aggregation_metadata_clean == true`
  - `exp2925.taxonomy_corrigendum_clean == true`
  - `exp2926.constraintbench_corrigendum_ready == true`
  - `exp2933.kan_cl_self_learning_ready == true`
- `exp2936` is intentionally ungated so the milestone can close honestly even
  when a branch is blocked.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` with GPU offload support.
- Mandated headline GGUFs available through `cached_sota_pair()`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Legacy Qwen3.5-0.8B or gemma-4-E4B-it models may appear only as fast CPU
  smoke tests and cannot be headline-result models.

Required for hardware tasks:

- KV260 reachable through `ssh kria`, with clean `.275` same-basis evidence:
  `results/experiment_2913_kv260_hardware_vs_cpu_speedup_claim_boundary_v1.json`.
- GateMate A1-EVB-2M attached through DirtyJTAG.
- OSS CAD Suite tools available through the discovered current path:
  `nextpnr-himbaechel --device CCGM1A1`, `gmpack`, `yosys`, and
  `openFPGALoader`.

Out of scope:

- Host-side KV260 SD-card manipulation.
- Any GateMate speedup claim before a board transcript and matched CPU basis.
- Extropic TSU/Z1/XTR-0 hardware claims.
- AMD XDNA/NPU tasks unless a later directive reopens them.
- Training a new EBT/NRGPT-scale model.

## Agent Routing

- Codex/gpt-5.5 is assigned to formulaic code, verifier, sampler, dataset, and
  GateMate build tasks.
- Claude Opus is reserved for hardware flash/timing coordination and capstone
  synthesis.
- Claude Sonnet remains the default for straightforward aggregation and
  planning tasks.
- Live LLM experiments must include one of the mandated SOTA local GGUFs in
  `MODEL_SPECS`.

## Acceptance Criteria

The milestone is successful if:

- `.275` is archived and `.276` activates cleanly.
- The metadata/provenance false positives around `exp2911`, `exp2921`, and
  `exp2922` are either corrected or honestly retired.
- ConstraintBench v2 reports syntax validity, feasibility, and optimality with
  non-identical denominators and a reproducibility checksum.
- GateMate either reaches a hashed n=16 bitstream through the corrected
  himbaechel flow or remains blocked with current-toolchain evidence.
- At least one continuous self-learning artifact (`exp2933`) reports utility,
  forgetting, and verifier-energy effects for a structural KAN/KAC-style update.
- Matrix v10 and the `.276` capstone preserve clean/flagged/blocked/pilot-only
  distinctions without upgrading rows by implication.
