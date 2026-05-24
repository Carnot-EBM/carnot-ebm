# Research Roadmap vNEXT: Milestone 2026.05.279

**Title:** DCCD Repair Replication + Solver-Frontier Formalization + GateMate Flash

**Planned:** 2026-05-24

**Previous milestone:** 2026.05.278

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.278 Proved

Milestone `.278` completed all scheduled tasks, but the capstone kept
`paper_ready=false`. The authoritative terminal artifact is
`results/experiment_2961_capstone_v278.json`.

- Code verifier operating policy is now clean: `exp2953` selected a default
  threshold with PPV 0.888889, recall 1.0, and false-accept rate 0.010135.
- GateMate constraints and an n=16 bitstream exist: `exp2955` materialized the
  constraints and `exp2956` built a timing-clean bitstream at 15.69 MHz.
- PolarFire scaled from 500 to 1000 clauses with hash/transcript evidence in
  `exp2958`.
- Taxonomy-guided code repair produced a positive small-N result
  (`pass@1_delta=0.25`, `false_accept_delta=-0.125`) but stayed flagged because
  the evaluated set was only four tasks and depended on flagged manifest inputs.
- FR-11 utility-gated replay improved held-out utility by 0.111859 without
  triggering rollback, but stayed flagged because it did not yet prove a
  non-tautological learning loop under stronger reset/forgetting controls.
- NL-to-Z3 execution was repaired, but formalization quality remained too weak:
  parseability and Z3 execution were both 0.083333, and solver-verified accuracy
  was 0.0.
- GateMate flash/timing smoke stayed blocked: `exp2957` ended with
  `blocked_board_not_detected`, so there is still no board flash, output hash,
  or post-flash timing claim.

## Three Biggest Gaps

### Gap 1: Positive Code Repair Evidence Is Too Small and Flagged

`.278` showed that taxonomy-guided repair can move pass@1, but the result is
not paper-grade. `.279` must replicate the result on at least 20 tasks, make the
structured-output path explicit, and use the mandated local SOTA GGUF models.
DCCD and BEAVER-style certificate fields should reduce schema failure without
turning verifier acceptance into another false-positive path.

### Gap 2: Exact Solver Reasoning Needs Skill Separation

The current NL-to-Z3 row proves Z3 can run, not that the local models can
formalize logic. `.279` should split formal reasoning into symbolization,
countermodel construction, validity assessment, and solver execution, then use
structured local GGUF proposals only after the exact-verifier corpus is clean.

### Gap 3: FR-11 and Hardware Both Need Non-Tautological Proof

FR-11 replay remains scheduler-level evidence until it survives reset, negative
control, and forgetting checks. GateMate remains materialized but not observed
on board. `.279` should produce one non-tautological self-learning artifact and
one hardware-contact/flash path; both must preserve claim boundaries.

## New Research Integrated

The 2026-05-24 post-`.278` sweep appended these items to
`research-references.md`:

- **DCCD** (arXiv:2603.03305): draft-then-constrain decoding is the direct
  structured-generation retry path for code repair and formalization.
- **LogicSkills and LLMEval-Logic** (arXiv:2602.06533, arXiv:2605.19597):
  exact-verifier logic work should be skill-labeled before live local GGUF
  retries.
- **Interwhen** (Microsoft Research, 2026): supports a partial-response monitor
  harness that checks verifiable properties during reasoning rather than only
  after final output.
- **BEAVER and `llguidance`**: deterministic prefix-bound/certificate ideas and
  local CFG/JSON constrained-output masks should be added to candidate
  manifests before another larger repair result is trusted.
- **KAN forgetting and KAN hardware complexity**: KAN memory is promising in
  low-dimensional algorithmic settings but still needs explicit forgetting
  controls and hardware-cost accounting.
- **CEM, EBT, ARM-as-EBM, Extropic/THRML, and Kona/Aleph**: useful theory and
  architecture context only; no `.279` experiment may claim native EBT,
  Extropic hardware, TSU, Kona, or Aleph-equivalent performance.

## Architecture Snapshot

```text
        +---------------------------------------------------------+
        | Phase A: constrained repair replication                 |
        |                                                         |
        | exp2962 archive/activate                               |
        | exp2963 DCCD repair protocol + manifest                 |
        | exp2964 gated SOTA DCCD repair replication              |
        | exp2965 BEAVER-style certificate audit                  |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase B: exact solver frontier                          |
        |                                                         |
        | exp2966 LogicSkills/LLMEval materializer                |
        | exp2967 gated SOTA NL-to-Z3 DCCD formalization          |
        | exp2968 Interwhen-style partial monitor harness         |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase C: continuous self-learning guards                |
        |                                                         |
        | exp2969 FR-11 non-tautological utility gate             |
        | exp2970 KAN forgetting guard + memory audit             |
        +---------------------------+-----------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        | Phase D: hardware contact and closeout                  |
        |                                                         |
        | exp2971 GateMate board-detection triage                 |
        | exp2972 gated GateMate flash/output-hash smoke          |
        | exp2973 cross-corpus matrix v13                         |
        | exp2974 capstone .279                                   |
        +---------------------------------------------------------+
```

## Phase Structure

### Phase A: Constrained Repair Replication

- `exp2962` archives `.278` and activates `.279`.
- `exp2963` turns DCCD into a local repair protocol: unconstrained draft,
  constrained repair, deterministic schema checks, and acceptance gates.
- `exp2964` runs the actual live SOTA replication on at least 20 code-repair
  tasks, gated on the protocol. It must use at least one of the mandated local
  SOTA GGUF models through the `cached_sota_pair()` pattern.
- `exp2965` adds BEAVER-style certificate fields over the repaired candidate
  manifests: prefix-closed constraints, explored/blocked frontier summaries,
  schema validity, and false-accept audit status. This is a bounded certificate
  audit, not a full BEAVER implementation.

### Phase B: Exact Solver Frontier

- `exp2966` materializes a LogicSkills/LLMEval-inspired exact-verifier mini set
  with reference Z3 formalizations and separate labels for symbolization,
  countermodel construction, and validity assessment.
- `exp2967` reruns NL-to-Z3 under DCCD-style structured formalization, gated on
  `exp2966`. Z3 remains the authority; LLM text is only a proposal.
- `exp2968` builds an Interwhen-style monitor harness that extracts and checks
  partial verifiable properties from code/logical outputs, recording coverage
  and latency without claiming full streaming verification.

### Phase C: Continuous Self-Learning Guards

- `exp2969` is the required continuous self-learning task. It reruns FR-11
  utility-gated replay with reset controls, random-replay controls, held-out
  utility, and a stricter forgetting guard.
- `exp2970` audits KAN/per-knot memory against frozen, eager, and adapter-style
  updates, using KAN forgetting findings to decide whether the FR-11 memory path
  should remain low-dimensional or be retired from high-dimensional claims.

### Phase D: Hardware Contact and Closeout

- `exp2971` diagnoses GateMate board detection and prepares a flash harness from
  the `.278` bitstream. It may report cable/tool/IDCODE status and precondition
  readiness only.
- `exp2972` flashes the `.278` GateMate bitstream and records an output hash
  only if `exp2971` proves board detection and bitstream integrity. No speedup,
  Boltzmann, or thermodynamic sampling claim is allowed.
- `exp2973` rebuilds cross-corpus matrix v13 with `.279` repair, solver,
  self-learning, KAN, monitor, and GateMate rows.
- `exp2974` closes `.279`, classifies clean/flagged/blocked rows, and recommends
  the next milestone.

## Dependency Graph

```text
exp2962

exp2963
  -> exp2964
  -> exp2965

exp2966
  -> exp2967

exp2968
exp2969
  -> exp2970

exp2971
  -> exp2972

exp2964
exp2965
exp2967
exp2968
exp2969
exp2970
exp2972
  -> exp2973

all artifacts
  -> exp2974
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2964` gates on `exp2963.dccd_repair_protocol_ready == true`.
- `exp2965` gates on `exp2963.dccd_repair_protocol_ready == true`.
- `exp2967` gates on `exp2966.logic_frontier_materialized == true`.
- `exp2970` gates on `exp2969.non_tautological_self_learning_ready == true`.
- `exp2972` gates on:
  - `exp2971.gatemate_board_detected == true`
  - `exp2971.bitstream_sha256_verified == true`
- `exp2973` gates on `exp2969.non_tautological_self_learning_ready == true`.
- `exp2974` is intentionally ungated so the milestone can close honestly even
  if a branch is blocked.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` or equivalent local GGUF runtime with GPU offload.
- Mandated headline GGUFs available through `cached_sota_pair()`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Legacy Qwen3.5-0.8B or gemma-4-E4B-it models may appear only as CPU
  smoke tests and cannot be headline-result models.

Required for exact-verifier tasks:

- Python `z3` import must succeed.
- Any local structured-output backend (`llguidance`, llama.cpp grammar, JSON
  schema validation) should be detected and reported rather than assumed.

Required for hardware tasks:

- GateMate A1-EVB-2M attached through DirtyJTAG.
- OSS CAD Suite path with `yosys`, `nextpnr-himbaechel --device CCGM1A1`,
  `gmpack`, and `openFPGALoader`.
- `.278` GateMate bitstream artifact:
  `results/experiment_2956_gatemate_n16_bitstream_build_v4.json`.

Out of scope:

- KV260 speedup, Boltzmann, or thermalization claims.
- Extropic TSU/Z1/XTR-0 hardware claims.
- Kona or Aleph performance claims.
- Native EBT training claims.
