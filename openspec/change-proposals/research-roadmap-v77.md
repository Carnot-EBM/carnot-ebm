# Research Roadmap v77: EnvPropagation Permanent Fix + Production Tier 2 + KV260 First Light

**Milestone:** 2026.04.77
**Status:** Planned
**Predecessor:** 2026.04.76 (research-roadmap-v76.md)
**Experiments:** 986–999 (14 experiments)

---

## What Milestone 2026.04.76 Proved

Milestone .76 met only 2 of 10 success criteria. The root cause was a single cascading failure:
**Exp 975 (EnvPropagationGuard) produced no result artifact**, blocking 6 downstream experiments
that gated on it. The failure pattern is identical to Exp 971 in .75: a missing `try/finally`
in the experiment script caused the artifact write to never execute when an error occurred.

**Successes in .76:**
- KV260 bitstream generated for first time in project history (Exp 982): Vivado 2025.2.1
  complete, bitstream at `output/carnot_ising_v4_bd/carnot_ising.bit`. Board not programmed
  — `kv260.local` DNS/network unreachable.
- Langevin SB (LSB) sampler deployed as default (Exp 983): 1.17x speedup, arXiv 2512.02323.
- arxiv scan added 5 new references (Exp 984).
- Preflight complete (Exp 974): manifest synced, SOTA models confirmed, Exp 906 diagnosed.

**Carry-forwards (unexecuted from .76, all gated on failed Exp 975):**
1. EnvPropagationGuard fix (Exp 975) — never produced artifact
2. SC-Energy Tier 2 test fix + deployment (Exp 976) — gated on 975
3. DualGPU fresh wiring validation (Exp 977) — gated on 975
4. Triple Integration E2E (Exp 978) — gated on 976
5. Fast-Path Probe Live GPU (Exp 979) — gated on 975
6. PPSEBM Live GPU Relay (Exp 981) — gated on 975
7. KAN MILP Violation Fix (Exp 980) — blocked by gate config bug (`op: ''`)

**Additional carry-forward:**
- KV260 board programming — bitstream ready, board network unreachable

---

## Architecture at Milestone Entry (2026.04.77)

```
  LLM Response
      |
      v
  [Tier 0a] CarnotThinkProbe        ← CPU CoT pre-filter (Exp 444)
      |
      v
  [Tier 0b] SpilledEnergyDetector   ← CPU logit-based (Exp 433/949 AUROC=1.0 synthetic)
      |                                  NOT YET live GPU validated (.76 Exp 979 blocked)
      v
  [Tier 0c] NUP Probe v4            ← CPU bigram dot product (Exp 523 AUC=1.0)
      |
      v
  [Tier 0d] HallucinationBasin      ← CPU finite-difference (Exp 521)
      |
      v
  [Tier 0e] HalluField              ← CPU advisory (Exp 571 AUC=0.97 synthetic)
      |
      v
  [Tier 1] SinkProbe                ← CPU attention sink (Exp 346-348)
      |
      v
  [Tier 2] SC-Energy OOD detector   ← JEPA retired (Exp 957); SC-Energy designated
      |                                  3 test failures block deployment (Exp 969/976)
      v
  [Tier 2.5] SymCodeVerifier        ← AUC=0.804 live (Exp 619)
      |
      v
  [Tier 2.6] HermesVerifierAdapter  ← CPU prototype (Exp 633, RETIRED Exp 641)
      |
      v
  [Tier 2.7] CausalReasoningVerifier ← causal_recall=0.36 (Exp 642)
      |
      v
  [Tier 3] IsingEBM + KAEMEnergy    ← KAEMEnergy: 11 formal violations unfixed (.76 Exp 980 blocked)
      |
      v
  [Repair] IterativeSelfRepair      ← Code: +10% (Exp 967); Math: RETIRED (Exp 963)
      |
      v
  [DualGPU] VerifyRepairPipeline    ← 1.96x benchmarked (Exp 932); wiring stale from .75
```

---

## The Three Biggest Gaps (PRD Vision vs Current State)

### Gap 1: EnvPropagationGuard blocks all live GPU work (2 consecutive milestones)

CARNOT_ env vars do not propagate to subprocesses when the conductor launches experiments.
This single failure blocked 6 of 10 .76 success criteria and an equal number in .75.
The .66 fix (Exp 855, RETRO-015 close) patches the in-session state but does not persist
across conductor session boundaries — each new session starts without the env vars.

**Root cause of the failure pattern:** Exp 975 never wrote an artifact because it lacked a
`try/finally` guard. When the script encountered an error during the fix implementation, the
artifact write was skipped. Exp 987 MUST have unconditional try/finally.

**PRD relevance:** FR-12 (Verifiable Reasoning) requires live end-to-end verification.
Until env propagation is permanent and reliable, all live GPU benchmarks are blocked.

### Gap 2: Production pipeline has known defects (SC-Energy 3 test failures + KAN 11 violations)

The SC-Energy Tier 2 OOD detector has been designated as the production Tier 2 since JEPA
was retired (Exp 957), but 3 pre-existing test failures (discovered Exp 969, unresolved in
.76 Exp 976 due to cascade block) prevent deployment. The KAEMEnergy (Tier 3) has 11 formal
MILP violations in monotonicity and boundary constraints (Exp 972), making energy scores
unreliable. A pipeline with known defects in its energy function cannot satisfy FR-12.

### Gap 3: KV260 board has bitstream but has never run hardware inference

The KV260 bitstream was generated in .76 (Exp 982, first time in project history). But the
board is unreachable at `kv260.local` (DNS/network issue). The board has been physically
present since 2026-04-20 (7 milestones). Hardware Ising latency has never been measured.
The Phase 2 hardware path (KV260 → D-Wave → Extropic Z1) cannot be validated until at
least one hardware latency number exists.

---

## Milestone Design

### Success Criteria (10)

1. EnvPropagationGuard fix persists across conductor session boundaries (Exp 987)
2. SC-Energy Tier 2 test failures fixed — 0 failures after fix (Exp 988)
3. SC-Energy wired as production Tier 2 OOD detector (Exp 988)
4. DualGPU VerifyRepairPipeline wiring confirmed with fresh timestamp (Exp 989)
5. Triple Integration cascade (SC-Energy + SpilledEnergy + ThinkPRM) E2E validated (Exp 990)
6. SpilledEnergy AUROC >= 0.70 on live GPU data (Exp 991)
7. KAN MILP violations fixed: n_violations == 0 after fix (Exp 992)
8. KV260 board programmed OR hardware latency measured (Exp 993)
9. PPSEBM cross-session memory validated on real GPU violations (Exp 994)
10. Retrospective written; ops/status.md and ops/changelog.md updated (Exp 999)

### Bonus Criteria (not in the 10 but valued if achieved)
- PCIB Tier 0f prototype deployed (Exp 995)
- GS-KAN energy tier implemented with AUC parity to standard KAN (Exp 996)
- NK-Optimizer KAEMEnergy training speedup confirmed (Exp 997)

---

## Phase Structure

### Phase 0: Governance + Critical Blocker Fix (2 experiments, no upstream gate)

**Exp 986: Preflight v27**
- Verify exclusion manifest covers 786/627/603/641 entries
- Validate ALL gated_on ops in .77 roadmap YAML are non-empty valid ops
- Confirm SOTA GGUF models present: Qwen3.6-35B-A3B-GGUF, Gemma-4-31B-it-GGUF, Gemma-4-26B-A4B-it-GGUF
- CRITICAL: verify Exp 980's gate bug (op='') is not repeated in .77 YAML

**Exp 987: EnvPropagationGuard Fix v2**
- Implement persistent state file: write CARNOT_ vars to `~/.carnot/conductor_state.sh`
- Update `scripts/experiment_template.py`: `EnvPropagationGuard.propagate()` sources the file
- Add try/finally to this script AND mandate it in experiment_template.py for all GPU experiments
- Produce artifact UNCONDITIONALLY (try/finally must wrap entire main body)
- Verify by running a subprocess that checks `echo $CARNOT_FORCE_LIVE` after new session start

### Phase 1: Production Tier 2 Deployment (2 experiments, gated on Exp 987)

**Exp 988: SC-Energy Tier 2 v3**
- Read Exp 969 artifact to find the 3 failing tests
- Fix the root cause (likely missing mock or import path)
- Wire SC-Energy as Tier 2 in VerifyRepairPipeline cascade
- Output: test_failures_fixed=True, tier2_wired=True, n_failing_tests_after=0

**Exp 989: DualGPU Pipeline v4**
- Wire DualGPURunner into VerifyRepairPipeline production path
- Produce fresh timestamped throughput result (NOT "deliverable already exists")
- Output: dualgpu_wired=True, throughput_ratio >= 1.5, timestamp (new .77 era)

### Phase 2: Pipeline Integration E2E (2 experiments, gated on Exp 988)

**Exp 990: Triple Integration E2E v3**
- Run SC-Energy + SpilledEnergy + ThinkPRM cascade end-to-end on 50 questions
- Produce fresh timestamped result (NOT "deliverable already exists")
- Output: cascade_validated=True, tier2_skip_rate > 0, timestamp

**Exp 991: Fast-Path Probe Live GPU v3**
- Run SpilledEnergy + NUP Probe on live GPU-inferred 50 CoT steps
- Measure AUROC on real violations
- Requires: CARNOT_FORCE_LIVE=1, SOTA GGUF model, try/finally on artifact write
- MODEL_SPECS must include unsloth/gemma-4-31B-it-GGUF or Qwen3.6-35B-A3B-GGUF
- Output: live_auroc >= 0.70, inference_mode='live_gpu'

### Phase 3: Hardware + EBM Quality (3 experiments)

**Exp 992: KAN MILP Violation Fix v2** (INDEPENDENT — no upstream gate)
- Read Exp 972 artifact: 11 violations (monotonicity + boundary in KAEMEnergy)
- Fix spline constraints in KAEMEnergy to satisfy MILP monotonicity and boundary conditions
- Re-run MILP verifier to confirm 0 violations
- Note: this experiment had a gate config bug (`op: ''`) in .76 that blocked it. .77 YAML
  has NO gate for this task — it is unconditionally independent.
- Output: n_violations_after_fix=0, kan_milp_verified=True

**Exp 993: KV260 Board Programming v3** (INDEPENDENT — no upstream gate)
- Bitstream is ready at `output/carnot_ising_v4_bd/carnot_ising.bit`
- Try IP discovery: `avahi-browse -t _ssh._tcp` or `nmap -sn 192.168.x.0/24`
- Try USB UART if network unavailable: check /dev/ttyUSB* after connecting board USB
- If board reachable: scp bitstream + run `dfx-mgr-client -load`
- If still unreachable: document exact error + steps for human-assisted programming
- Measure CPU baseline latency as lower bound (target hardware: < 100 µs)
- Output: board_programmed=True OR human_action_required=True, hardware_latency_us

**Exp 994: PPSEBM Live GPU Relay v2** (gated on Exp 991 live_violations_collected)
- Load real violations from Exp 991 artifact
- Run 5 PPSEBM sessions on the real violations (not CPU synthetic)
- Measure sessions_with_new_templates (target >= 3)
- MODEL_SPECS must include a SOTA GGUF for live inference in session simulation
- Output: live_relay_confirmed=True, sessions_with_new_templates, inference_mode='live_gpu'

### Phase 4: Research Frontier (3 experiments, all INDEPENDENT)

**Exp 995: PCIB Hallucination Tier 0f** (arXiv 2601.15652)
- Implement PCIB (Predictive Coding + Information Bottleneck) hallucination signals
- Extract entity-uptake and falsifiability scores from Gemma4-E4B-it on FoVer corpus
- Train a KAN energy function on PCIB features
- Compare AUROC vs existing Tier 0 probes on 57-pair FOVER corpus
- Output: pcib_auroc, vs_nup_probe_delta, tier0f_viable=True/False

**Exp 996: GS-KAN Energy Tier** (arXiv 2512.09084)
- Implement GS-KAN (Sprecher-type shared basis functions) in python/carnot/models/kan.py
- Train on FoVer 57-pair corpus
- Compare: AUROC vs standard KAN, parameter count, estimated LUT usage
- Target: AUROC parity with < 50% of standard KAN's parameter count
- Output: gs_kan_auroc, param_reduction_pct, lut_estimate, auroc_vs_standard_delta

**Exp 997: NK-Optimizer KAEMEnergy** (arXiv 2512.18921)
- Replace Adam optimizer with Newton-Kaczmarz in KAN energy training
- Train on 57-pair FoVer corpus and 500-pair Z3-expanded corpus
- Compare training time and AUROC vs Adam baseline
- Output: nk_vs_adam_speedup, nk_auroc, adam_auroc, epochs_to_auc095

### Phase 5: Close (2 experiments)

**Exp 998: arxiv Research Scan + research-references.md update**
- Search for recent papers on: EBMs for verification, KAN variants, FPGA Ising, guided
  decoding, continual learning for constraints, hardware-accelerated sampling
- Add any papers not already in research-references.md

**Exp 999: Milestone 2026.04.77 Retrospective**
- Read all .77 experiment artifacts
- Compute success criteria results
- Update ops/status.md, ops/changelog.md
- Write results/experiment_999_milestone_retro_77.json

---

## Dependency Graph

```
Exp 986 (Preflight v27)
  └── Exp 987 (EnvPropagation Fix v2)
        ├── Exp 988 (SC-Energy Tier 2 v3)
        │     └── Exp 990 (Triple Integration E2E v3)
        ├── Exp 989 (DualGPU Pipeline v4)
        └── Exp 991 (Fast-Path Probe Live GPU v3)
              └── Exp 994 (PPSEBM Live GPU Relay v2)

Exp 992 (KAN MILP Fix v2)     — INDEPENDENT (no gate — .76 gate bug fixed)
Exp 993 (KV260 Board v3)      — INDEPENDENT
Exp 995 (PCIB Tier 0f)        — INDEPENDENT
Exp 996 (GS-KAN Energy)       — INDEPENDENT
Exp 997 (NK-Optimizer KAEM)   — INDEPENDENT
Exp 998 (arxiv Scan)          — INDEPENDENT
Exp 999 (Retro)               — reads all above
```

---

## Hardware Requirements

| Experiment | Requires GPU | Notes |
|------------|-------------|-------|
| Exp 986-987 | No | Infrastructure only |
| Exp 988-990 | No | CPU wiring + integration tests |
| Exp 991 | Yes (RTX 3090) | Live inference, SOTA GGUF |
| Exp 992 | No | MILP solver, CPU |
| Exp 993 | No | Board programming (Vivado host CPU) |
| Exp 994 | Yes (RTX 3090) | Live inference for self-learning |
| Exp 995-997 | No | CPU training + benchmarks |
| Exp 998-999 | No | Web research + doc update |

GPU experiments: CARNOT_FORCE_LIVE=1, sg render -c '...' for GPU group access.
SOTA models must be pre-downloaded by Exp 986 preflight.

---

## arxiv Findings Incorporated

- **arXiv 2601.15652** (PCIB hallucination detection, January 2026) → Exp 995
- **arXiv 2512.09084** (GS-KAN parameter efficiency, December 2025) → Exp 996
- **arXiv 2512.18921** (NK-optimizer KAN training, December 2025) → Exp 997
- **arXiv 2604.16430** (HalluSAE phase-transition energy, April 2026) → filed to research-references.md for .78
- **arXiv 2602.11364** (DiffuTruth Semantic Energy, February 2026) → filed for .78
- **arXiv 2603.06875** (Stochastic Attention Langevin, March 2026) → filed for .79
- **arXiv 2604.20052** (ALMC-ODE multimodal sampling, April 2026) → filed for .78

---

## Governance Discipline

### Critical process failure from .76 to fix in .77

1. **try/finally mandate:** Every experiment script that writes an artifact MUST wrap the
   entire main body in try/finally, with artifact write in the finally block. This is the
   direct root cause of Exp 975 (and Exp 971 in .75) failing silently.

2. **Gate op validation:** Every `gated_on` field in the YAML must have a non-empty op
   string from the supported set: `==, !=, >, >=, <, <=, in, not_in, contains, not_contains`.
   The empty op `op: ''` that blocked Exp 980 in .76 must never appear.

3. **Independent experiments must have NO gate:** Exp 992 (KAN MILP Fix) and Exp 993 (KV260)
   are genuinely independent and must not be gated on EnvPropagationGuard. Gating independent
   work on infrastructure experiments is the cascade failure pattern we must eliminate.

### Exclusion manifest status entering .77

| Experiment | Consecutive Appearances | Manifest Status |
|------------|------------------------|-----------------|
| Exp 786 | 17 → 0 in .76 | In manifest (added Exp 952 + confirmed .76 Exp 974) |
| Exp 627 | 17 → 0 in .76 | In manifest (added Exp 974) |
| Exp 603 | 17 → 0 in .76 | In manifest (YAML has entry from .58) |
| Exp 641 | 7 → 0 in .76 | In manifest (added Exp 952, confirmed .76 Exp 974) |
| Exp 906 | 4 → 0 in .76 | Root cause diagnosed in Exp 974: 50q_scale_latency |

The .76 retro confirms slowest-5 composition changed for the first time — legacy carryovers
absent from .76 run set. .77 must maintain this discipline.

---

## Decentralization Review (CLAUDE.md rules 1-7)

- Rule 1 (local-first): All GPU experiments use SOTA GGUF models via llama.cpp — no closed-weight dependency
- Rule 2 (closed frontier optional): No closed-weight model calls in this milestone
- Rule 3 (mirroring): GS-KAN (Exp 996) energy tier publishable to HF + IPFS if AUC target met
- Rule 4 (multiple surfaces): SC-Energy Tier 2 wiring (Exp 988) applies to pipeline API surface
- Rule 5 (hardware portability): KV260 track (Exp 993) + NK/GS-KAN (Exp 996/997) advance
  hardware-portable KAN inference. GS-KAN's parameter reduction targets KV260 BRAM budget.
- Rule 6 (data minimization): No closed-weight model calls in this milestone
- Rule 7 (no vendor abstractions in core): SC-Energy wired via SamplerBackend protocol

All rules satisfied. No decentralization-degraded features introduced.

---

## Self-Learning Tier Coverage

Per research-program.md requirement, at least one experiment per milestone must advance
the continuous self-learning architecture (Tiers 1-4):

- **Exp 994 (PPSEBM Live GPU Relay v2)**: Tier 2 self-learning — cross-session constraint
  memory PPSEBM validated on real GPU violations (not CPU synthetic). This is the first
  real-data validation of Tier 2 constraint memory since the FR-11 relay at Milestone .33.
  **Satisfies the mandatory self-learning experiment requirement.**
