# Research Roadmap vNEXT: 2026.07.467

## Milestone Title

LOCAL SOTA RUNTIME REPAIR + EXACT-VERIFIER PIVOT + GOVERNED FR-11

## Why This Milestone Exists

Milestone `2026.07.466` completed its conductor pass, but it did not deliver the intended process-
verifier decision. The central blocker was operational, not theoretical: the mandated SOTA GGUF model
files were cached, but no live completion/logprob endpoint was reachable. That gate-blocked uPRM and
VPR, then cascaded into a skipped decision gate and capstone.

What `.466` proved:

1. **SOTA files are present, but the endpoint is not.** `exp5071` found the three required GGUF model
   files locally, but `completion_endpoint_ready`, `logprob_endpoint_ready`, and
   `top_logprob_or_confidence_ready` were all false. More uPRM planning is wasteful until this is fixed.
2. **Replay-only guided decoding is not enough.** `exp5075` found DCCD equal to unguided and worse than
   rerank-only (`delta_dccd_vs_rerank = -0.150`). The next constrained-generation experiment must change
   mechanism, not resample the same DCCD surface.
3. **The tool-first cascade is a cost lead, not a headline win.** `exp5076` had an accuracy point
   estimate over judge-only, but the confidence interval touched zero and the artifact correctly refused
   a Pareto headline.
4. **FR-11 memory remains unsafe without governance.** `exp5077` improved dev but lost held-out
   (`heldout_delta = -0.050`) and rolled back. `exp5078` retired blind replay and promote-on-dev memory.
5. **Hardware continuity is real but still non-headline.** `exp5079` kept KV260 and PolarFire reachable;
   GateMate remained undetected; no speedup claim is justified.
6. **KAN formalization has a live foothold.** `exp5080` verified a tiny PWA/MILP property for a KAEM/KAN
   unit. This should scale with binary-count/error-budget telemetry before any architectural claim.
7. **The capstone path itself needs repair.** `exp5081`/`exp5082` were skipped by dead gate structure.
   The next capstone must run over available artifacts and record blockers rather than disappearing.

The purpose of `.467` is therefore: **repair local SOTA runtime truth, run a final gated uPRM retry only
if live logprobs are real, and shift the primary verifier-moat research toward exact solvers, structured
constraint masks, KAN/MILP verification, and governed continuous self-learning.**

## Three Biggest Gaps Versus The PRD

### Gap 1: Runtime Provenance Blocks PRD-Grade Verifier Claims

PRD FR-12 requires verifiable reasoning with auditable provenance. `.466` had cached GGUFs but no live
logprob endpoint, so the most important process-verifier tasks never executed. `.467` must treat the
llama.cpp/GGUF endpoint as a first-class experiment with transcript, model path, endpoint health,
completion, logprob/top-logprob telemetry, and duration/provenance fields.

### Gap 2: Free-Form Verifier Moat Has Hit Diminishing Returns

MuSR-style scalar verifier work is bounded, DCCD lost to rerank-only, and process rewards are blocked
until telemetry is repaired. The PRD vision points to verifiable reasoning, not judge imitation. Fresh
research reinforces the pivot: p-bit guided CDCL keeps SAT correctness in CDCL/Z3, STATIC makes finite
constraint masks hardware-friendly, and Logical Intelligence's Kona/Aleph framing puts formal checkers
at the center of correctness-critical systems.

### Gap 3: Continuous Self-Learning Has No Safe Memory Lifecycle Yet

PRD FR-11 requires autonomous self-learning. Carnot has repeatedly shown that blind memory/replay can
harm held-out performance. The next FR-11 work must use budgeted memory, provenance, poisoning/staleness
guards, on-policy replay from the current system, and rollback. A small positive is acceptable; an honest
no-promote ledger is also useful if it narrows the mechanism.

## Fresh Research Folded In

Added to `research-references.md` before this plan under `V467-PLANNER-REFERENCES`:

- `arXiv:2606.25313` - Programmable Probabilistic Computer with 1,000,000 p-bits.
- `arXiv:2605.04033` - p-bit guided CDCL using Ising consensus assumptions.
- `arXiv:2602.22647` and `github.com/youtube/static-constraint-decoding` - STATIC CSR constrained
  decoding.
- `arXiv:2503.14495` / OpenReview `sM5QDzIg3j` - temporal consistency for process error detection.
- `arXiv:2505.19706` / `declare-lab/PathFinder-PRM` - hierarchical error-aware PRMs.
- `arXiv:2512.22322` - SmartSnap proactive evidence seeking.
- `arXiv:2506.11442` / OpenReview `q56ZI1Co43` - ReVeal self-verifying code agents.
- `arXiv:2606.25115` - budget-curated memory for on-device agents.
- `arXiv:2605.29495` - on-policy replay for continual SFT.
- `arXiv:2510.17281` and `arXiv:2512.18950` - MemoryBench and hierarchical procedural memory.
- `arXiv:2606.18206`, `arXiv:2605.11011`, and `arXiv:2603.12248` - EBT/ARM citation-lineage pressure:
  fixed-point/looped reasoning and energy-based fine-tuning.
- `arXiv:2605.09186` and `arXiv:2602.06737` - solver-aware MIP agents and KAN PWA/MILP verification.
- `arXiv:2602.04200` and `arXiv:2601.17094` - sparse Potts constraints and EBM world-model separation.
- Extropic XTR-0/TSU and Logical Intelligence Kona/Aleph updates.

## Architecture For .467

```text
                         research-references.md
                                  |
                                  v
                +-----------------------------------+
                | Exp5083 archive .466 truth        |
                +-----------------------------------+
                                  |
                                  v
                +-----------------------------------+
                | Exp5084 SOTA/current refs audit   |
                +-----------------------------------+
                                  |
                                  v
                +-----------------------------------+
                | Exp5085 llama.cpp GGUF endpoint   |
                +-----------------------------------+
                         |                    |
             logprobs?   |                    | endpoint optional
                         v                    v
        +-----------------------------+   +------------------------------+
        | Exp5086 uPRM logprob cache  |   | Exp5088 temporal consistency |
        +-----------------------------+   +------------------------------+
                         |
                         v
        +-----------------------------+
        | Exp5087 uPRM process retry  |
        +-----------------------------+

        +-----------------------------+   +------------------------------+
        | Exp5089 p-bit CDCL bridge   |   | Exp5090 STATIC CSR masks     |
        +-----------------------------+   +------------------------------+
                         |                    |
                         v                    v
        +-----------------------------+   +------------------------------+
        | Exp5091 KAN PWA/MILP scale  |   | Exp5092 governed FR-11       |
        +-----------------------------+   +------------------------------+

        +-----------------------------+
        | Exp5093 hardware continuity |
        +-----------------------------+
                         |
                         v
        +--------------------------------------------------------------+
        | Exp5094 ungated capstone over all available artifacts         |
        +--------------------------------------------------------------+
```

## Phases

### Phase 0: Transition, SOTA Ingestion, And Runtime Repair

Experiments: `exp5083`, `exp5084`, `exp5085`

Archive `.466` honestly, append/verify the current source set, and bring up a local llama.cpp/GGUF
endpoint for at least one mandated SOTA model. `exp5085` is the load-bearing experiment: it must either
produce live completion and logprob/top-logprob telemetry or emit an actionable blocker with exact binary,
port, model, CUDA, and server-log evidence.

### Phase 1: Process-Verifier Recovery Without Dead Gates

Experiments: `exp5086`, `exp5087`, `exp5088`

Run uPRM only behind structured runtime gates:

- `exp5086` runs only if `exp5085.logprob_endpoint_ready == true`.
- `exp5087` runs only if `exp5086.logprob_cache_ready == true`.

In parallel, `exp5088` tests temporal-consistency process verification as a logprob-free fallback. It
may use live local SOTA completions if `exp5085` made them available, but it must still emit a useful
diagnostic without pretending blocked logprobs exist.

### Phase 2: Exact Verifiers And Structured Constraints

Experiments: `exp5089`, `exp5090`, `exp5091`

Shift the main verifier-moat surface to domains with objective checkers:

- `exp5089` prototypes p-bit/Ising-guided CDCL where SAT/Z3 owns correctness.
- `exp5090` replaces DCCD with STATIC-style CSR masks for finite verifier schemas and structured output.
- `exp5091` scales the KAN PWA/MILP bridge from a tiny property to small multi-unit properties with
  binary-count, error-budget, and solver-time telemetry.

### Phase 3: Governed Self-Learning, Hardware Continuity, And Capstone

Experiments: `exp5092`, `exp5093`, `exp5094`

`exp5092` is the required continuous self-learning experiment: budget-curated memory plus on-policy
replay under rollback, poisoning, non-forgetting, and held-out guards. `exp5093` preserves hardware
continuity across KV260, GateMate, and PolarFire without speedup claims. `exp5094` is deliberately
ungated so the milestone always ends with a reconciled decision artifact over whatever ran.

## Dependency Graph

```text
exp5083
  -> exp5084
      -> exp5085
          -> exp5086
              -> exp5087
          -> exp5088
      -> exp5089
      -> exp5090
      -> exp5091
      -> exp5092
      -> exp5093

exp5083, exp5084, exp5085, exp5086, exp5087, exp5088,
exp5089, exp5090, exp5091, exp5092, exp5093
  -> exp5094
```

Structured `gated_on` entries are used only where the conductor can skip wasted calls safely:

- `exp5086` waits for `exp5085.logprob_endpoint_ready == true`.
- `exp5087` waits for `exp5086.logprob_cache_ready == true`.

The capstone has no structured gate. It must run and record blocked/missing upstream artifacts rather
than reproducing the `.466` decision-gate cascade.

## Hardware Requirements

- **Dual RTX 3090 / CUDA host:** required for llama.cpp GGUF endpoint bring-up and any live local SOTA
  inference. Use `scripts/experiment_template.py::cached_sota_pair()` patterns and resolved local `.gguf`
  files; never call `AutoTokenizer.from_pretrained()` on `-GGUF` repos.
- **Mandated local GGUFs for any LLM task:**
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **KV260:** SSH-only checks via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no host
  `/dev/mmcblk*` preconditions. UIO/register reads must be safe and transcript-backed.
- **GateMate A1 / DirtyJTAG:** detect/terminal-state triage only unless IDCODE and bitstream path are
  proven in the artifact.
- **PolarFire:** SSH reachability and hash-verified dispatch/precheck; no flash/timing claim without
  transcript.
- **Extropic/TSU:** architecture/simulation reference only; no local TSU hardware target exists.

## Falsifiable Gates

1. A runtime result is positive only if it records the concrete local model path, binary/server command,
   endpoint URL, sample prompt, completion, logprob/top-logprob evidence, and duration/provenance.
2. uPRM is positive only if it uses real SOTA GGUF logprob telemetry and beats tuned self-consistency with
   paired statistics; otherwise it is bounded or blocked.
3. Temporal-consistency PRM is positive only if first-error/process classification improves over one-pass
   verifier and self-consistency baselines without model-identity leakage.
4. p-bit CDCL is positive only if solver correctness remains exact and CDCL conflict/propagation effort
   improves on a declared instance family; distribution-sensitive nulls are acceptable.
5. STATIC CSR decoding is positive only if it improves schema validity or latency against CPU trie and
   rerank-only baselines without hurting accuracy.
6. KAN PWA/MILP scale is positive only if solver size, error budgets, and property status are all reported;
   a tiny-only proof is not a scale claim.
7. FR-11 is positive only if held-out delta is non-negative, non-forgetting passes, poisoning/provenance
   guards pass, and rollback prevents harmful promotion.
8. Hardware artifacts may claim only the reachability, register, dispatch, and timing evidence actually
   captured in transcripts.

## Expected Deliverables

- `research-roadmap-next.yaml` with 12 conductor tasks in execution order.
- `research-references.md` updated with the `.467` source set before experiment design.
- `results/experiment_5083_archive_466_activate_467.json` through
  `results/experiment_5094_capstone_v467.json` when the conductor runs.
- Capstone reconciliation should update `ops/status.md`, `ops/changelog.md`,
  `_bmad/traceability.md`, and any affected OpenSpec capability docs after execution.
- `_bmad/architecture.md` is stale relative to July hardware/runtime reality; if `.467` changes runtime
  or solver architecture, reconcile it during or immediately after the capstone.
