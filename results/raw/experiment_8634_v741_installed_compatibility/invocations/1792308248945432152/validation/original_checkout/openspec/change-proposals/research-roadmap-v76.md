# Research Roadmap v76: EnvPropagation Fix + Production Tier 2 Deployment + KAN Repair

**Milestone:** 2026.04.76
**Status:** Planned
**Predecessor:** 2026.04.75 (research-roadmap-v75.md)
**Experiments:** 974–985 (12 experiments)

---

## What Milestone 2026.04.75 Proved

Milestone .75's primary success criterion was process discipline: zero experiments blocked by
missing prior_failures. **The criterion was partially met: 4 of 10 success criteria were satisfied.**

Successes:
- Math repair RETIRED: sota_ceiling_confirmed (repair_delta=0.0 with Gemma4-31B), no re-proposal
- Code repair 100q viable: code_repair_delta=+0.10 (+10%), DebugRepair hypothesis step adds zero
  net value (hypothesis_contribution=0.0) — simplify to plain iterative repair
- Symbolic-KAN deployed to HuggingFace + IPFS dual-distribution confirmed
- PPSEBM cross-session memory plateau broken: sessions_with_new_templates >= 3 achieved on CPU
- KAN formal verification completed: 11 violations found (monotonicity and boundary conditions)
- Process discipline: zero prior_failures gate blocks in .75 conductor cycle

Failures / gaps entering .76:
1. **EnvPropagationGuard** (P0): CARNOT_ env vars not propagating to subprocesses — blocked ALL
   live GPU experiments. This is the root cause behind Exp 964/965/966/969/971 not producing
   fresh result files. Explains many multi-milestone live GPU misses.
2. **SC-Energy Tier 2 wiring** (P0): Exp 969 discovered 3 pre-existing test failures with unknown
   root cause. SC-Energy cannot be deployed until these are fixed.
3. **Stale pipeline validation** (P1): Exp 965 (Triple Integration) and Exp 966 (DualGPU) returned
   "deliverable already exists" — no fresh .75-era validation. New timestamped results required.
4. **KV260 board programming** (P1): No artifact JSON produced in .75. Vivado synthesis, bitstream
   generation, and dfx-mgr-client board programming remain unexecuted on the physical board.
5. **KAN violations unfixed** (P1): 11 monotonicity/boundary violations in KAEMEnergy remain.
   The energy model cannot be trusted for production verification until violations are resolved.

---

## Architecture at Milestone Entry (2026.04.76)

```
  LLM Response
      |
      v
  [Tier 0] SpilledEnergy fast reject      ← CPU, logit-based, no GPU (Exp 949 AUROC=1.0 CPU)
      |                                        NOT YET live GPU validated (Exp 964 blocked)
      v
  [Tier 1] ArithmeticExtractor / VeriCoT  ← CoACEExtractor (Python eval), Z3 formal steps
      |
      v
  [Tier 2] SC-Energy OOD detector         ← JEPA retired (Exp 957); SC-Energy designated
      |                                        but 3 test failures block deployment (Exp 969)
      v
  [Tier 3] IsingEBM + KAEMEnergy          ← 11 formal violations found (Exp 972) — unfixed
      |
      v
  [Repair] IterativeSelfRepair            ← Code: +10% (Exp 967); Math: RETIRED (Exp 963)
      |
      v
  [DualGPU] VerifyRepairPipeline          ← 1.96x benchmarked (Exp 932); wiring stale .75
```

---

## The Three Biggest Gaps (PRD Vision vs Current State)

### Gap 1: Pipeline broken at every tier boundary (EnvPropagation + test failures)
The pipeline cannot deliver end-to-end verified results from a live GPU because:
(a) CARNOT_ env vars don't reach subprocesses — live inference never starts
(b) SC-Energy Tier 2 has 3 test failures — cannot be deployed without regression risk
(c) KAEMEnergy has 11 formal violations — energy scores may be wrong

**Why it matters:** Every live benchmark is unreliable until these are fixed. The PRD's FR-12
(verifiable reasoning) requires a trusted end-to-end pipeline. A pipeline with known violations
in its energy function and broken subprocess inheritance cannot satisfy FR-12.

### Gap 2: Self-learning confined to CPU synthetic data (FR-11 relay incomplete)
PPSEBM cross-session memory plateau is broken on CPU synthetic data (Exp 970). But the
self-learning loop requires real errors from live GPU inference to learn anything meaningful.
Until the live GPU pipeline works (Gap 1), Tier 2 self-learning (PPSEBM) cannot be validated
on real data. The FR-11 relay (Exp 443, JEPA AUC 0.457→0.571 on 57 real pairs) was the last
confirmed real-data relay — that was at Milestone .33 and the relay has not advanced since.

**Why it matters:** FR-11 (Autonomous Self-Learning Loop) is the core differentiator that
separates Carnot from static EBM frameworks. Without real-data learning, we have infrastructure
but no actual self-improvement.

### Gap 3: Hardware acceleration path stalled (KV260 unboarded 7 milestones after arrival)
KV260 board arrived 2026-04-20 (7 milestones ago). Vivado synthesis has been attempted and
blocked repeatedly. The E-MVL sparse v4 RTL (Exp 958, 27K LUTs within budget) is ready.
No bitstream has ever been generated; no hardware latency has been measured.

**Why it matters:** The hardware acceleration path (KV260 → D-Wave → Extropic Z1) is the
path to FR-07 performance targets (NFR-01: 10x Rust throughput) and ultimately Phase 2.
A board that has been present for 7 milestones without producing a hardware latency number
is a project credibility issue. Nature Comms 2025 (thermodynamic computing SPU) validates
the physics; we need to validate the engineering.

---

## Milestone Design

### Success Criteria (10)

1. EnvPropagationGuard implemented and validated (Exp 975)
2. SC-Energy Tier 2 test failures diagnosed and fixed (Exp 976)
3. SC-Energy wired as production Tier 2 OOD detector (Exp 976)
4. DualGPU VerifyRepairPipeline wiring confirmed with fresh timestamped result (Exp 977)
5. Triple Integration cascade (SpilledEnergy → ThinkPRM → SC-Energy) validated E2E (Exp 978)
6. Fast-path probes validated on live GPU: SpilledEnergy AUROC >= 0.70 (Exp 979)
7. KAN MILP violations fixed: n_violations_found == 0 after fix (Exp 980)
8. KV260 bitstream generated OR hardware latency measured (Exp 982)
9. PPSEBM cross-session memory validated on live GPU-inferred violations (Exp 981)
10. Retrospective written; ops/status.md and ops/changelog.md updated (Exp 985)

### Phase Structure

**Phase 0: Governance + Critical Blocker (2 experiments)**
- Exp 974: Preflight v26 — manifest enforcement + Exp 906 retirement + SOTA model verify
- Exp 975: EnvPropagationGuard Fix — subprocess env propagation repair

**Phase 1: Production Tier 2 Deployment (2 experiments, gated on Exp 975)**
- Exp 976: SC-Energy Tier 2 v2 — diagnose 3 test failures + deploy Tier 2 OOD detector
- Exp 977: DualGPU Pipeline v3 — fresh timestamped DualGPU wiring validation

**Phase 2: Pipeline Integration (2 experiments, gated on Exp 976/977)**
- Exp 978: Triple Integration v2 — SC-Energy + SpilledEnergy + ThinkPRM cascade (fresh)
- Exp 979: Fast-Path Probe Live GPU v2 — SpilledEnergy + ThinkPRM + DRIFTProbe (real data)

**Phase 3: Hardware + EBM Quality (2 experiments)**
- Exp 980: KAN MILP Violation Fix — repair 11 monotonicity/boundary violations in KAEMEnergy
- Exp 981: PPSEBM Live GPU Relay — cross-session memory on real GPU violations
- Exp 982: KV260 Board Programming v2 — Vivado bitstream + board programming

**Phase 4: Research Frontier (2 experiments)**
- Exp 983: Langevin SB Parallelizable Boltzmann Sampler (arXiv 2512.02323)
- Exp 984: Arxiv Research Scan + research-references.md update

**Phase 5: Close (1 experiment)**
- Exp 985: Milestone 2026.04.76 Retrospective

### Dependency Graph

```
Exp 974 (Preflight)
  └── Exp 975 (EnvPropagation Fix)
        ├── Exp 976 (SC-Energy Tier 2 v2)
        │     └── Exp 978 (Triple Integration v2)
        │           └── Exp 979 (Fast-Path Probe Live GPU v2)
        │                 └── Exp 981 (PPSEBM Live GPU Relay)
        └── Exp 977 (DualGPU Pipeline v3)

Exp 980 (KAN MILP Fix)    — independent
Exp 982 (KV260 Board v2)  — independent
Exp 983 (Langevin SB)     — independent
Exp 984 (Arxiv Scan)      — independent
Exp 985 (Retro)           — reads all above
```

---

## Hardware Requirements

| Experiment | Requires GPU | Notes |
|------------|-------------|-------|
| Exp 974-975 | No | Infrastructure only |
| Exp 976-978 | No | CPU wiring + integration tests |
| Exp 979 | Yes (RTX 3090) | Live inference validation |
| Exp 980 | No | MILP solver, CPU |
| Exp 981 | Yes (RTX 3090) | Live inference for self-learning data |
| Exp 982 | No | Vivado synthesis (host CPU) |
| Exp 983 | No | CPU sampling benchmark |
| Exp 984 | No | Web research only |
| Exp 985 | No | Doc update |

GPU experiments require: CARNOT_FORCE_LIVE=1, sg render -c '...' for GPU group access,
unsloth/gemma-4-31B-it-GGUF pre-downloaded.

---

## arxiv Findings Incorporated

- **arXiv 2512.02323** (LSB parallelizable Boltzmann sampler) → Exp 983 (Langevin SB)
- **arXiv 2505.15960** (generalizable PRM via formal annotations) → reference for future
  SC-Energy v3 corpus expansion (Z3-labeled 500+ pairs) in .77
- **Nature Comms 2025** (thermodynamic computing SPU) → validates Phase 2 hardware path;
  confirms D-Wave → Extropic Z1 escalation is grounded in published physics

---

## Governance Discipline

### Retirement mandates entering .76

| Experiment | Consecutive Appearances | Status | Action |
|------------|------------------------|--------|--------|
| Exp 786 | 16 | In manifest (Exp 952) | Verify manifest enforced at dispatch site |
| Exp 627 | 16 | NOT in manifest | Add to exclusion_manifest.yaml in Exp 974 |
| Exp 603 | 16 | In manifest (partial) | Verify manifest enforced at dispatch site |
| Exp 641 | 6 | In manifest (Exp 952) | Verify manifest enforced at dispatch site |
| Exp 906 | 3 | NOT in manifest | Diagnose root cause; add to manifest if unfixable |

### Process improvements targeted for .76

- **EnvPropagationGuard** (Exp 975): mechanical fix, not documentation
- **SOTA model pre-download**: in Exp 974 preflight, download all 3 SOTA GGUFs before GPU experiments run
- **Experiment count cap**: target <= 662 (hold .75 level, avoid creep back above 700)
- **Zero slowest-5 composition change prevention**: if .76 shows same slowest-5, escalate to human

---

## Decentralization Review (CLAUDE.md rules 1-7)

- Rule 1 (local-first): All experiments use SOTA GGUF models via llama.cpp — no closed-weight dependency
- Rule 2 (closed frontier optional): GPU experiments use unsloth GGUF models, local inference
- Rule 3 (mirroring): Symbolic-KAN published to HF + IPFS (Exp 968). KV260 bitstream (Exp 982)
  will use IPFS mirroring per same pattern.
- Rule 4 (multiple surfaces): Pipeline, CLI, MCP, HTTP REST all maintained
- Rule 5 (hardware portability): KV260 track (Exp 982) + Langevin SB (Exp 983) advance
  hardware-portable sampling
- Rule 6 (data minimization): No closed-weight model calls in this milestone
- Rule 7 (no vendor abstractions in core): SC-Energy (Exp 976) wired via SamplerBackend protocol

All rules satisfied. No decentralization-degraded features introduced.
