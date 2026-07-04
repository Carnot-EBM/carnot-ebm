# Research Roadmap vNEXT: 2026.07.478

Status: proposed for activation after milestone `2026.07.477`

Milestone title: GAP-1 Registry Promotion, GAP-4 Provenance Repair, Continuous Memory, and ARC Skill Gates

## Inputs Read

This plan is grounded in:

- `research-program.md`
- `_bmad/prd.md`
- `_bmad/architecture.md`
- `ops/status.md`
- `ops/changelog.md`
- `research-complete.yaml`
- `research-roadmap.yaml` for completed milestone `2026.07.477`
- `openspec/change-proposals/`
- `ops/conductor-log.md`
- `research-references.md`
- `research-hardware-wishlist.md`
- `CLAUDE.md`

The plan also incorporates the 2026 SOTA update appended to `research-references.md` on 2026-07-04.

## What 2026.07.477 Proved

Milestone `.477` closed with mixed but useful signal:

- GAP-1 set-search hardening produced a real positive held-out effect:
  `heldout_pass2_mean = 0.189584`, paired delta CI approximately `[0.0231, 0.0604]`.
  The result was not promoted because the downstream gate expected a bare boolean while the artifact stored
  `gap1_hardened_positive.value = true`, and because the best subset was not yet stable enough to promote
  without an explicit registry decision.
- GAP-4 candidate expansion and scale validation are quarantined. `exp5211` and `exp5212` were adversarially
  flagged for tautology/protocol problems; no GAP-4 headline result can use those artifacts until provenance,
  model-use records, seeds, and protocol pass fields are repaired.
- The MMLU-Pro hidden-state verifier path is retired. The chunk/layer sweep did not beat simple controls and
  produced an explicit `retire_mmlu_hidden_state_path = true` result.
- Continuous self-learning exists but is still shallow. `exp5214` wrote verifier-memory promotions and
  rollbacks, satisfying the milestone mandate, but it is not yet integrated as typed memory used by future
  verifier or ARC decisions.
- ARC did not bank a new live-path level. PAW amortization was not viable in the bounded pilot, and frontier
  continuity plus landmark decomposition produced `reproducible_total_levels_delta = 0`.
- Hardware continuity is intact without speedup claims: KV260 and PolarFire are reachable; GateMate remains
  blocked at the physical/JTAG layer.
- Verifier authenticity improved through registry flags, but some modules remain headline-ineligible until
  they perform real verification rather than uncertainty routing or naming-only checks.

## Three Biggest Gaps to PRD Vision

1. **Verifier evidence is not yet registry-grade.**
   The PRD needs verifiable reasoning components, but `.477` left GAP-1 positive evidence unpromoted and
   GAP-4 artifacts quarantined. `.478` must convert one positive thread into a registry decision and must
   prevent future candidate pools from passing without concrete provenance fields.

2. **The ARC live agent still lacks a learning loop that changes behavior.**
   The architecture has a useful offline twin and 69 reproducible registry levels, but the live path is not
   yet improving hidden-game discovery. `.478` should stop trying another static solve variant and instead
   add process-rubric and provenance gates that identify whether the live agent is failing at skill
   selection, skill following, skill composition, or reflection.

3. **Continuous self-learning is a ledger, not an operating system.**
   The research program calls for autonomous directed self-learning. The next step is typed memory with
   promotion/rollback policies, retention tests, and explicit consumers in verifier/ARC decisions. AutoMem,
   Multi-Head Recurrent Memory, and SkillCoach make this a natural `.478` experiment.

## SOTA Findings Incorporated

The SOTA scan added these experiment-driving references:

- VerIbmc / Neuro-Symbolic Software Verification (`arXiv:2606.16886`): local open-weight LLMs plus ESBMC
  feedback for invariant synthesis.
- VERGE (`arXiv:2601.20055`): semantic routing to SMT/ATP plus minimal correction-set localization.
- P-GCD (`arXiv:2606.01926`) and STATIC (`arXiv:2602.22647`): constrained decoding should be protocolled and
  hardware-conscious, not untracked generation.
- ProvenanceGuard (`arXiv:2607.01236`) and source-aware factuality (`arXiv:2606.18037`): generated artifacts
  must carry source/model/provenance records.
- AutoMem (`arXiv:2607.01224`), Multi-Head Recurrent Memory Agents (`arXiv:2607.01523`), SkillCoach
  (`arXiv:2607.01874`), and AgenticSTS (`arXiv:2607.02255`): continuous self-learning should use typed memory
  and process rubrics rather than broad self-distillation.
- Optimal KAN abstractions (`arXiv:2602.06737`) and GRS-KAN (`arXiv:2607.01449`): KAN-style energy modules
  need certifiable abstractions before they become verifier substrate.
- One-million p-bit probabilistic computer (`arXiv:2606.25313`), large-scale Ising FPGA decomposition
  (`arXiv:2602.15985`), and Extropic TSU/XTR-0 updates: hardware planning should focus on sampler workload
  boundaries and reproducible correctness hashes, not speedup claims.

## Architecture Target

```text
                 Local SOTA GGUFs
     Qwen3.6-35B-A3B / Gemma-4-31B / Gemma-4-26B-A4B
                         |
                         v
              +----------------------+
              | Candidate Proposers  |
              | - GAP-4 constrained  |
              | - VerIbmc invariant  |
              | - ARC live skill use |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Deterministic Gates  |
              | - Z3/SMT/ESBMC      |
              | - protocol schemas   |
              | - provenance checks  |
              | - registry promotion |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Typed Self-Learning  |
              | - constraint memory  |
              | - provenance memory  |
              | - failure memory     |
              | - skill/rubric memory|
              +----------+-----------+
                         |
                         v
              +----------------------+
              | ARC / Verifier Users |
              | - E3AgentPolicy      |
              | - OfflineSolver twin |
              | - KAEM/KAN modules   |
              | - hardware samplers  |
              +----------------------+
```

The milestone keeps the PRD architecture direction: local models propose, deterministic verifiers decide,
memory updates are promoted only through auditable gates, and hardware remains a correctness/reachability
track until there is a stable sampler workload.

## Phase Plan

### Phase 0 - Activate and Refresh

Tasks:

- `exp5220-archive-477-activate-478`
- `exp5221-sota-ingestion-v478`

Goal: close `.477` truthfully, pre-stage `.478`, and retry SOTA/citation discovery without deriving claims
from Semantic Scholar rate-limited results.

### Phase 1 - Registry and Provenance Repair

Tasks:

- `exp5222-gap1-gate-field-and-registry-promotion-v478`
- `exp5223-gap4-flagged-pool-authenticity-audit-v478`
- `exp5224-gap4-canonical-pool-builder-v478`
- `exp5225-gap4-clean-scale-validation-gated-v478`

Goal: promote or explicitly block the GAP-1 set verifier from the real positive `.477` result, then repair
GAP-4 so future validation uses a canonical candidate pool with model IDs, seeds, durations, source paths,
and protocol pass fields. The GAP-4 validation task is gated on the canonical-pool builder.

### Phase 2 - Verifier Feedback and Continuous Memory

Tasks:

- `exp5226-veribmc-local-solver-feedback-pilot-v478`
- `exp5227-continuous-self-learning-multihead-memory-v478`

Goal: test a local SOTA GGUF plus deterministic solver-feedback loop on a small code/formalization set, then
upgrade the self-learning ledger into typed memory that can preserve promotions, rollbacks, provenance, and
process rubrics across tasks.

### Phase 3 - ARC Live Path, KAN Certificate, and Hardware Continuity

Tasks:

- `exp5228-arc-provenance-skill-rubric-gate-v478`
- `exp5229-arc-gated-live-levelup-from-rubric-v478`
- `exp5230-kan-milp-verifier-certificate-v478`
- `exp5231-hardware-continuity-pbit-boundary-v478`

Goal: improve ARC live-agent diagnosis without duplicate/offline solve claims, run one gated live-path patch
only if the rubric exposes a real intervention, test a small KAN/PWA/MILP certificate path, and keep hardware
continuity bounded to reachability, hashes, and sampler-boundary planning.

### Phase 4 - Capstone and Reconciliation

Task:

- `exp5232-capstone-v478`

Goal: reconcile specs, status, changelog, references, memory, and exclusion/provenance decisions. The capstone
must exclude flagged or gated artifacts from headline claims.

## Dependency Graph

```text
exp5220 archive/activate
  |
  +--> exp5221 SOTA ingestion
  |
  +--> exp5222 GAP-1 registry decision
  |
  +--> exp5223 GAP-4 authenticity audit
          |
          v
      exp5224 GAP-4 canonical pool builder
          |
          v
      exp5225 GAP-4 clean validation

exp5221 SOTA ingestion
  |
  +--> exp5226 VerIbmc local solver feedback
  |
  +--> exp5227 multi-head memory
          |
          +--> exp5228 ARC skill rubric
                    |
                    v
                exp5229 ARC gated level-up

exp5221 SOTA ingestion
  +--> exp5230 KAN/MILP certificate
  +--> exp5231 hardware p-bit boundary

all non-blocked tasks
  |
  v
exp5232 capstone
```

Structured gates:

- `exp5225` runs only when `exp5224.gap4_canonical_pool_usable == true` and
  `exp5224.canonical_pool_n >= 120`.
- `exp5229` runs only when `exp5228.arc_skill_rubric_usable == true` and
  `exp5228.recommended_live_patch_available == true`.

## Model and Inference Requirements

Every experiment that uses an LLM must use at least one mandated local SOTA GGUF model:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Experiment prompts must direct implementers to use `scripts/experiment_template.py` and the
`cached_sota_pair()` pattern when local LLM inference is needed. Legacy small models may appear only as CPU
smoke tests and cannot support headline claims.

LLM-bearing tasks in this milestone:

- `exp5224-gap4-canonical-pool-builder-v478`
- `exp5226-veribmc-local-solver-feedback-pilot-v478`
- `exp5229-arc-gated-live-levelup-from-rubric-v478` if the live patch invokes a model proposer

## Hardware Requirements

- Dual RTX 3090 CUDA box: needed for local GGUF generation/evaluation in GAP-4 and VerIbmc-style pilots.
- CPU-only path: required for audits, registry edits, KAN certificate scaffolding, memory ledger tests, and
  fast smoke tests.
- KV260: SSH-only reachability and hash smoke, no speedup claim.
- PolarFire: reachability/hash smoke only, no terminal workload claim unless already available.
- GateMate A1 EVB: do not claim progress without physical/JTAG change; record `0xffffffff` IDCODE if still
  blocked.
- RX 7900 XTX / ROCm: not a headline dependency for `.478`.

## No-Go Rules

- Do not modify `research-roadmap.yaml` during planning.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not rerun the retired MMLU-Pro hidden-state verifier path.
- Do not reopen Phase-D external generated-text scoring.
- Do not claim ARC solves from offline BFS, source reading, hand-built game adapters, or outer-loop reverse
  engineering.
- Do not use `exp5211` or `exp5212` GAP-4 artifacts as headline evidence unless a later task repairs and
  validates a canonical replacement pool.
- Do not claim hardware speedups in `.478`.
- Do not create root scratch scripts.

## Success Criteria

Minimum useful milestone:

- A registry-grade GAP-1 decision is made from the `.477` positive artifact, with tests and docs.
- GAP-4 has a canonical provenance-clean pool or an explicit retirement/block reason.
- Continuous self-learning advances from static ledger to typed memory with promotion/rollback tests.
- ARC produces either one live-path gated improvement attempt with valid `solve_provenance` or a concrete
  process-rubric diagnosis that prevents another blind level-up attempt.
- Hardware continuity remains truthful and reproducible.
- Capstone reconciles `openspec/`, `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`, and result
  artifacts.

Headline-eligible stretch:

- GAP-1 enters the verifier registry with a reproducible holdout artifact.
- GAP-4 clean validation beats the prior baseline on the canonical pool without adversarial flags.
- VerIbmc-style local solver feedback shows positive uplift over deterministic-only and LLM-only baselines.
- ARC banks a new level through `live_agent_self_discovery`.
