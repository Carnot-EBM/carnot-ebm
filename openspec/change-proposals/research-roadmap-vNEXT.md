# Carnot Research Roadmap vNEXT: Repair Corrigendum, FR-11 Held-Out Learning, GateMate Output Contract

**Created:** 2026-05-25
**Milestone:** 2026.05.284
**Status:** Planned
**Supersedes:** 2026.05.283 ("Claim Repair v2 + Feasibility-Gated Self-Learning + GateMate IO Boundary")
**Execution file:** `research-roadmap-next.yaml`
**Primary evidence inputs:** `results/experiment_3012_*.json` through `results/experiment_3025_*.json`

## What .283 Proved

| Area | Experiments | Finding |
|---|---:|---|
| SOTA GGUF readiness | 3013 | Local SOTA telemetry preflight is available for the mandated GGUF set. |
| Repair diagnosis | 3014-3016 | Syntax/schema taxonomy and Cactus-style acceptance gates produced a SOTA repair rerun with `pass_at_1_delta=0.375`, `syntax_failure_rate_delta=0`, `schema_failure_rate_delta=0`, `false_accept_delta=0`, and `tautology_gate_clean=true`. |
| Validator tree | 3017-3018 | NSVIF-style validator expansion and a BEAVER-style frontier certificate are available, but unresolved bounds remain visible. |
| FR-11 self-learning | 3019-3020 | The feasibility-channel diagnostic remains flagged for tautology risk, while the DVI verifier-feedback controller itself completed cleanly. |
| GateMate | 3021-3023 | GateMate RTL/CCF work isolated the real blocker: host-visible output pinout/transport is not ready. The downstream flash smoke and SSQA path correctly gate-skipped. |
| Matrix/capstone | 3024-3025 | Matrix v17 is complete (`clean=40`, `flagged=29`, `blocked=10`, `gated_skipped=3`, `missing=1`). Capstone is ready but `paper_ready=false`; next focus is repair corrigendum plus GateMate output contract. |

**Main conclusion:** .283 created useful positive evidence, but not a paper-ready or PRD-ready milestone. The next milestone must distinguish true methodology defects from adversarial false positives, promote FR-11 only with held-out nonforgetting controls, and resolve the GateMate output contract before any hardware smoke claim.

## Three Biggest Gaps

1. **Evidence promotion gap:** repair deltas are positive, but matrix/capstone still carry adversarial and methodology flags. The system needs a corrigendum artifact that replays source transcripts, duration/provenance fields, model specs, seeds, checksums, and claim boundaries before any headline repair claim.

2. **FR-11 autonomy gap:** the DVI controller is promising, but the feasibility channel still has tautology risk. FR-11 needs held-out replay, information-asymmetric checking, negative controls, and nonforgetting stress before it can count as continuous self-learning instead of cached self-confirmation.

3. **Hardware observability gap:** GateMate cannot proceed from RTL to smoke until the output path is host-visible. The next step is not speedup; it is a concrete pinout/output contract, simulation evidence, and a flash/smoke gate that fails fast when the contract is still unavailable.

## New Research Integrated

The 2026-05-25 sweep added these planning inputs to `research-references.md`:

| Source | Use in .284 |
|---|---|
| MARCH (arXiv:2603.24579) | Information-asymmetric checker/proposer split for repair and FR-11 audits. |
| Draft-Conditioned Constrained Decoding (arXiv:2603.03305) | Preserve unconstrained draft intent before syntax/schema constrained repair acceptance. |
| STATIC vectorized trie (arXiv:2602.22647) | Future accelerator-ready mask design; only accounting/design reference in .284. |
| Hard linear constraints with decision rules (OpenReview, NeurIPS 2025) | Feasible-by-construction framing for FR-11 promotion controls. |
| Clip-and-Verify (OpenReview, NeurIPS 2025) | Separate verified, irrelevant, unresolved, and fallback-only validator regions. |
| Self-Distillation Enables Continual Learning (arXiv:2601.19897) | Held-out and nonforgetting requirements for continuous self-learning. |
| KAN-CL (arXiv:2605.12306) | Per-locality nonforgetting probe design before any new KAN training. |
| FPGA Ising decomposition (arXiv:2602.15985) | Reinforces GateMate host-visible decomposition/output contract as the near-term hardware deliverable. |
| Extropic/Logical Intelligence public updates | Architecture context only; no borrowed hardware or benchmark claims. |

## Architecture Snapshot

```
                         ┌──────────────────────────────┐
                         │ .283 source artifacts         │
                         │ 3013-3025 JSON + logs         │
                         └──────────────┬───────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ Repair Evidence  │          │ FR-11 Controller │          │ GateMate Output  │
│ Corrigendum      │          │ Held-Out Replay  │          │ Contract         │
│                  │          │                  │          │                  │
│ - provenance     │          │ - DVI feedback   │          │ - pinout audit   │
│ - duration       │          │ - asym checker   │          │ - RTL/CCF shim   │
│ - model specs    │          │ - nonforgetting  │          │ - flash smoke    │
│ - DCCD/MARCH     │          │ - neg controls   │          │ - SSQA gate      │
└────────┬─────────┘          └────────┬─────────┘          └────────┬─────────┘
         │                              │                              │
         ▼                              ▼                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Validator Frontier / Claim Boundary                                           │
│ verified rows, unresolved rows, fallback-only rows, blocked hardware rows      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Matrix v18 + Capstone .284                                                   │
│ paper_ready only if repair evidence is clean, FR-11 is non-tautological,      │
│ and GateMate blocker is either resolved or explicitly bounded.                │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Phase Structure

### Phase A: Evidence Corrigendum and Repair Promotion (Exp3026-Exp3031)

Goal: turn .283's positive repair deltas into clean, bounded evidence or explicitly retire the claim path.

- Exp3026 archives .283 and activates .284 without modifying `research-roadmap.yaml`.
- Exp3027 audits adversarial flags and writes a methodology corrigendum manifest.
- Exp3028 reruns or reconstructs SOTA repair evidence under clean metadata and information-asymmetric checking.
- Exp3029 decides the repair promotion boundary from actual artifacts, not optimism.
- Exp3030 tightens validator frontier bounds using Clip-and-Verify-style region separation.
- Exp3031 tests draft-conditioned constrained repair on a tiny SOTA panel to see whether intent preservation improves without false accepts.

### Phase B: FR-11 Held-Out Continuous Self-Learning (Exp3032-Exp3033)

Goal: promote self-learning only if it survives held-out replay, nonforgetting, and negative controls.

- Exp3032 replays DVI verifier feedback on held-out exact traces with independent checker inputs.
- Exp3033 stress-tests nonforgetting and drift, using KAN-CL/SDFT as design references but not claiming weight learning unless actually performed.

### Phase C: GateMate Output Contract and SSQA Boundary (Exp3034-Exp3037)

Goal: resolve the host-visible output blocker or convert it into an exact operator-action artifact.

- Exp3034 audits GateMate pinout/toolchain/output choices and decides the contract.
- Exp3035 implements and simulates the RTL/CCF output shim only if the contract is ready.
- Exp3036 flashes and reads host-visible output only if simulation passes.
- Exp3037 writes the SSQA bounded RTL/PnR or explicit gate artifact.

### Phase D: Synthesis (Exp3038-Exp3039)

Goal: close the milestone with a matrix and capstone that do not hide blocked or flagged work.

- Exp3038 builds matrix v18 with no top-level live-model metadata.
- Exp3039 writes the .284 capstone and decides whether .285 should be paper-promotion, GateMate operator action, or FR-11 expansion.

## Dependency Graph

```
exp3026 archive
   └─ exp3027 methodology-corrigendum
        ├─ exp3028 SOTA repair clean-methodology rerun
        │    └─ exp3029 repair promotion boundary
        │         └─ exp3038 matrix v18
        ├─ exp3030 validator frontier corrigendum
        │    └─ exp3038 matrix v18
        └─ exp3031 draft-conditioned constrained repair panel
             └─ exp3038 matrix v18

exp3032 FR-11 held-out DVI replay
   └─ exp3033 FR-11 nonforgetting stress
        └─ exp3038 matrix v18

exp3034 GateMate output contract
   └─ exp3035 GateMate output shim sim
        └─ exp3036 GateMate flash smoke
             └─ exp3037 SSQA bounded RTL/PnR or gate artifact
                  └─ exp3038 matrix v18

exp3038 matrix v18
   └─ exp3039 capstone .284
```

## Hardware Requirements

| Requirement | Experiments | Notes |
|---|---:|---|
| Dual RTX 3090 / local CUDA GGUF cache | 3028, 3031 | Any live LLM task must use at least one mandated SOTA local GGUF: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy small models are smoke tests only. |
| CPU-only artifact aggregation | 3026, 3027, 3029, 3030, 3032, 3033, 3038, 3039 | Aggregation artifacts must not include top-level `model_specs`, `target_model`, CUDA, or GGUF fields. |
| GateMate A1 + DirtyJTAG | 3034-3037 | Required command: `openFPGALoader -c dirtyJtag --detect`; board target remains `-b olimex_gatemateevb`. No speedup claim without host-visible output transcript. |
| Extropic TSU / Z1 | none | Architecture context only; no local access assumed. |
| KV260 / PolarFire / NPU | none | Out of scope for .284 unless an operator separately reopens those tracks. |

## Acceptance Criteria

- `research-roadmap-next.yaml` validates with the roadmap schema and prior-failure linter.
- Every rerun-scope task includes complete `prior_failures` entries with `retire_if_same_verdict: true`.
- Every natural-language gate has a matching structured `gated_on` block using supported operators.
- Every live LLM experiment includes the mandated SOTA GGUF `MODEL_SPECS`.
- Exp3032 or Exp3033 satisfies the required continuous self-learning coverage.
- GateMate tasks either produce host-visible output artifacts or a precise blocked artifact with the missing pinout/operator action.
- Matrix v18 and capstone .284 preserve flagged, blocked, gated-skipped, projection-only, pilot-only, and missing rows honestly.

## Failed-Experiment Rerun Compliance

This milestone intentionally carries forward unresolved scope from .283:

- Repair rerun lineage: exp2977, exp2991, exp3003, exp3014-exp3016.
- Validator frontier lineage: exp3017-exp3018.
- FR-11 lineage: exp2983, exp2995, exp3007, exp3019-exp3020.
- GateMate/SSQA lineage: exp2984, exp2996, exp3008, exp3021-exp3023.
- Matrix/capstone lineage: exp3011, exp3024-exp3025.

The YAML declares `prior_failures` for those scopes. If a task repeats the same verdict after the stated change, it should retire the scope instead of becoming another carry-over.

## Out of Scope

- Editing `scripts/research_conductor.py`.
- Modifying `research-roadmap.yaml` during planning.
- Publishing, blog posts, public docs, README refreshes, or external submission.
- Claiming GateMate, Extropic, KV260, or NPU speedups without host-visible hardware transcripts.
- Treating small legacy GGUF smoke tests as headline SOTA evidence.
