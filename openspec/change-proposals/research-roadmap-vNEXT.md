# Research Roadmap vNEXT - Milestone 2026.05.285

**Title:** GateMate Output Unblock + Repair Flag Hygiene + Governed FR-11 Self-Learning
**Created:** 2026-05-25
**Status:** Planned
**Supersedes:** 2026.05.284 "Repair Corrigendum + FR-11 Held-Out Learning + GateMate Output Contract"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.284 Proved

Milestone `.284` completed its planned diagnostic arc but did not produce a
paper-ready capstone. The authoritative capstone is
`results/experiment_3039_capstone_v284.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `repair_claim_status=bounded`
- `fr11_self_learning_status=controller_only_promotable`
- `gatemate_status=blocked_pinout_missing_bounded`
- `ssqa_status=gate_skipped_bounded_no_performance_claim`
- `next=2026.05.285_gatemate_output_contract_repair_flag_cleanup`

The positive result is narrow but useful: `exp3028` produced a clean SOTA repair
candidate row over 24 tasks, and `exp3033` promoted FR-11 only as a
controller-side self-learning result with nonforgetting and negative-control
checks. The blocked result is equally important: `exp3034` showed that GateMate
cannot advance without a physical output contract and host reader command.

| Area | `.284` result | `.285` consequence |
| --- | --- | --- |
| SOTA repair | Clean rerun candidate from `exp3028`, but `exp3029` kept promotion bounded | Clean methodology and matrix/capstone flag hygiene must happen before headline promotion |
| Validator frontier | `exp3030` split verified, irrelevant, unresolved, and fallback-only rows | Add exact SMT/SAT feedback and avoid collapsing unresolved into clean |
| FR-11 | `exp3033` was controller-only promotable with held-out and negative-control checks | Add governed self-learning metrics: family contradictions, rollback, delayed regression |
| GateMate | `exp3034` blocked on missing output pinout and host read command | Produce an operator-ready contract package, then gate RTL/flash work structurally |
| SSQA | `exp3037` remained gate-skipped because GateMate had no host-visible output | Keep SSQA bounded unless host-visible smoke passes |
| Matrix/capstone | `exp3038` and `exp3039` remained bounded | Run matrix v19 after repair, FR-11, and hardware gates resolve or skip cleanly |

## Three Biggest Gaps To PRD Vision

1. **Evidence hygiene gap.** The PRD requires verifiable reasoning claims, but
   `.284` still left repair promotion bounded by adversarial/methodology
   aggregation flags. `.285` must distinguish real blockers from
   aggregation-substrate false positives, then either promote or keep the repair
   row bounded with exact citations.

2. **Continuous self-learning gap.** FR-11 is no longer a smoke test, but it is
   still controller-only. The next useful step is not model-weight training; it
   is governed self-learning over related constraint families with rollback,
   non-regression, and solver-grounded edit rights.

3. **Hardware observability gap.** GateMate toolchain presence is not enough.
   Carnot needs a host-visible output contract before any FPGA timing, SSQA, or
   hardware-accelerated sampler claim. `.285` should make this an explicit
   operator contract and gate all downstream work on it.

## New Research Integrated

The post-`.284` sweep was added to `research-references.md` before this
milestone was designed. The findings that materially shape `.285` are:

| Finding | Source | Milestone use |
| --- | --- | --- |
| LLM-42 / verified speculation | arXiv:2601.17768 | Add transcript fingerprint and replay discipline for local GGUF repair evidence |
| VERGE / MCS feedback | arXiv:2601.20055 | Convert verifier failures into exact correction subsets |
| SMT solver distillation | OpenReview ICLR 2026 Workshop | Separate NL-to-SMT translation validity from solver-execution validity |
| SATQuest | OpenReview ICLR 2026 Workshop and arXiv:2509.00930 | Use objective SAT/SMT families for self-learning and cross-format checks |
| Governed self-improvement | OpenReview ICLR 2026 Workshop | Add contradiction graph, rollback, and delayed-regression metrics to FR-11 |
| Graph Energy Matching | arXiv:2603.23398 | Keep as medium-term graph-energy context; do not add a new backend yet |
| Ontology-constrained neural reasoning | arXiv:2604.00555 | Use semantic-routing language for matrix claim classes |

## Architecture Snapshot

```text
                      +-------------------------------+
                      |  Mandated local SOTA GGUFs     |
                      |  Qwen3.6-35B-A3B, Gemma dense |
                      |  Gemma MoE                    |
                      +---------------+---------------+
                                      |
                                      v
+----------------------+     +-------------------------+     +----------------------+
| Repair evidence      | --> | Transcript fingerprint  | --> | Repair promotion    |
| exp3028 clean row    |     | verified-speculation    |     | boundary + matrix   |
+----------------------+     +-------------------------+     +----------------------+

+----------------------+     +-------------------------+     +----------------------+
| Validator frontier   | --> | SMT/SAT exact feedback  | --> | Governed FR-11      |
| verified/unresolved  |     | MCS-like correction set |     | self-learning loop  |
+----------------------+     +-------------------------+     +----------------------+

+----------------------+     +-------------------------+     +----------------------+
| GateMate A1-EVB      | --> | Output pinout + host    | --> | RTL sim -> flash    |
| toolchain present    |     | reader contract         |     | smoke -> SSQA gate  |
+----------------------+     +-------------------------+     +----------------------+

                                      |
                                      v
                      +-------------------------------+
                      | Cross-corpus matrix v19       |
                      | Capstone .285                 |
                      +-------------------------------+
```

## Phase Plan

### Phase A - Archive, Flag Hygiene, Deterministic Repair Evidence

Tasks: `exp3040`-`exp3043`

- Archive `.284` and stage `.285`.
- Audit matrix/capstone adversarial flags and separate real blockers from
  aggregation false positives.
- Reconcile the repair promotion boundary using `exp3028`, `exp3029`,
  `exp3038`, and `exp3039`.
- Add a verified-speculation-style transcript fingerprint preflight using at
  least one mandated local GGUF when live LLM inference is needed.

Exit condition: repair is either a clean promotable candidate with reproducible
metadata or remains bounded with exact blockers.

### Phase B - Solver-Guided Verification and Governed Self-Learning

Tasks: `exp3044`-`exp3047`

- Add an exact SMT/SAT validator-tree upgrade with correction-set evidence.
- Define FR-11 governance boundaries from the `.284` controller-only result.
- Run a solver-feedback self-learning loop with family holdouts, nonforgetting,
  rollback, and negative controls.
- Probe KAN locality/nonforgetting only as a bounded controller/locality result.

Exit condition: FR-11 can be promoted only as governed controller-side
self-learning unless a task actually produces stronger evidence.

### Phase C - GateMate Output Contract and SSQA Gate

Tasks: `exp3048`-`exp3051`

- Convert the `exp3034` blocker into an operator-ready output contract package.
- Gate RTL/CCF simulation on the contract being ready.
- Gate flash smoke on the simulation passing.
- Gate SSQA readback eligibility on a host-visible smoke transcript.

Exit condition: GateMate either reaches host-visible output evidence or remains
explicitly bounded without consuming downstream agent turns.

### Phase D - Matrix and Capstone

Tasks: `exp3052`-`exp3053`

- Build cross-corpus matrix v19 from `.284` and `.285` artifacts.
- Write a capstone that can be paper-ready only if repair, FR-11, GateMate, and
  SSQA claims are clean under the new matrix.

Exit condition: capstone reports `paper_ready=true` only if every promoted claim
has source artifacts, methodology, and claim boundaries.

## Dependency Graph

```text
exp3040 archive
  |
  v
exp3041 flag hygiene
  |----------------------.
  v                      v
exp3042 repair boundary  exp3043 transcript fingerprint

exp3044 SMT/SAT validator exactness
  |
  v
exp3046 solver-feedback self-learning
  ^
  |
exp3045 FR-11 governance
  |
  v
exp3047 KAN locality probe (gated on exp3046)

exp3048 GateMate output contract
  |
  v
exp3049 RTL/CCF shim sim
  |
  v
exp3050 host-visible flash smoke
  |
  v
exp3051 SSQA readback eligibility

exp3042, exp3043, exp3046, exp3047, exp3051
  |
  v
exp3052 matrix v19
  |
  v
exp3053 capstone .285
```

Structured conductor gates are present for the hard dependencies:

- `exp3042` and `exp3043` gate on `exp3041.flag_hygiene_ready`.
- `exp3046` gates on `exp3044.validator_tree_exactness_ready` and
  `exp3045.fr11_governance_ready`.
- `exp3047` gates on `exp3046.fr11_solver_feedback_ready`.
- `exp3049` gates on `exp3048.gatemate_output_contract_ready`.
- `exp3050` gates on `exp3049.gatemate_shim_sim_passed`.
- `exp3051` gates on `exp3050.gatemate_host_visible_smoke_passed`.
- `exp3053` gates on `exp3052.matrix_v19_ready`.

`exp3052` is intentionally not gated on hardware success; it must be able to
record bounded and gate-skipped rows.

## Hardware Requirements

| Resource | Required by | Requirement | Claim boundary |
| --- | --- | --- | --- |
| Dual RTX 3090 CUDA host | `exp3043`, optional live LLM in `exp3044` | At least one mandated GGUF loadable through the repo's SOTA cache path | No SOTA headline if only legacy smoke models run |
| GateMate A1-EVB-2M | `exp3048`-`exp3051` | Physical output pinout plus host reader command before flash smoke | No latency, sampler, Boltzmann, or speedup claim without host-visible transcript |
| KV260 | none in `.285` | Available but not part of this milestone | Do not use host SD-card preconditions |
| PolarFire | none in `.285` | Available but out of scope | No PolarFire claims |
| Extropic TSU/XTR-0 | none | Public context only | No access or performance implication |

## Acceptance Criteria

1. `research-roadmap-next.yaml` validates against schema and prior-failure
   linters.
2. Every task has a concrete JSON deliverable path.
3. Every prompt includes CONTEXT, EXISTING CODE TO READ FIRST, TASK, and
   CONCRETE STEPS.
4. Every prompt ends with: `Run command, 'Do NOT push. Do NOT modify scripts/research_conductor.py.'`
5. All LLM-using tasks name at least one mandated local SOTA GGUF in
   `MODEL_SPECS`.
6. `exp3046` satisfies the mandatory continuous self-learning requirement.
7. GateMate downstream tasks use structured `gated_on` fields so blocked
   hardware contracts skip before expensive agent calls.
8. Prior-failure metadata includes `retire_if_same_verdict: true` for all
   carry-forward scopes.
9. The matrix records clean, flagged, bounded, blocked, gated-skipped,
   projection-only, missing, and retired rows separately.
10. The capstone does not promote repair, FR-11, GateMate, or SSQA beyond the
    evidence in matrix v19.

## Failed-Experiment Rerun Compliance

The milestone intentionally revisits scopes from `.284`. Every carry-forward
task in `research-roadmap-next.yaml` includes `prior_failures` with the four
mandatory fields:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

The GateMate chain avoids referencing retired upstream experiments in `requires`
fields. Instead, it uses same-milestone structured gates.

## Out Of Scope

- No modification to `research-roadmap.yaml`.
- No modification to `scripts/research_conductor.py`.
- No public submission, arXiv action, PyPI action, or external publication.
- No Extropic, Kona, KV260, or PolarFire performance claim.
- No model-weight self-learning claim unless an experiment actually trains and
  verifies model weights, which this milestone does not plan.
- No hardware speedup, Boltzmann sampling, or SSQA performance claim without
  host-visible GateMate output.
