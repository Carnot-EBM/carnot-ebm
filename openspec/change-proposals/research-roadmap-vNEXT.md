# Research Roadmap vNEXT: 2026.07.468

## Milestone Title

EXACT-VERIFIER SCALE-UP + EVIDENCE ENERGY + FORMAL FR-11

## Why This Milestone Exists

Milestone `2026.07.467` completed and produced a useful decision: the cleanest progress came from
objective, exact-verifier surfaces, while runtime/process-verifier work remained operationally fragile.
The next milestone should not spend its center of gravity on another free-form verifier retry. It should
scale the exact-verifier path, add evidence-relative energy checks, repair local SOTA runtime provenance
only enough to unblock future work, and make continuous self-learning contract-governed instead of
memory-accumulation-driven.

What `.467` proved:

1. **Exact verification is the current positive path.** `exp5091` produced a clean KAN/PWA/MILP property
   proof with `property_holds=true`, `solver_status=optimal`, 6 binary variables, 43 constraints, and no
   adversarial flag. This is the first surface worth scaling.
2. **Runtime/process-verifier claims are still blocked by provenance.** `exp5085` reported completion and
   logprob readiness but was flagged because the run was too short for a live inference claim. `exp5086`
   then blocked on endpoint/cache state, and `exp5087` gate-blocked. `.468` needs a clean substrate
   artifact, not another downstream uPRM promise.
3. **Constrained generation needs semantic controls.** `exp5090` found a STATIC CSR mask win on latency
   and validity, but the artifact was flagged as tautological and too short. The next constrained
   decoding task must audit semantic distortion and no-op validity, not simply show syntactic masks.
4. **p-bit/CDCL was correctness-preserving but not yet useful.** `exp5089` preserved exact SAT authority
   but produced no effort win. Future Ising work should change representation or hardware telemetry,
   not repeat the same pairwise assumption heuristic.
5. **FR-11 memory remains safe but inert.** `exp5092` correctly refused to promote budgeted/on-policy
   memory when held-out and non-forgetting deltas were zero. The next self-learning experiment needs
   formal update contracts and richer evidence snapshots.
6. **Hardware continuity exists but no acceleration claim is justified.** `exp5093` kept KV260 and
   PolarFire paths alive, but GateMate detection still failed and no board-level speedup was proven.
7. **The milestone decision is an exact-verifier pivot.** `exp5094` recorded
   `complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked`, excluding flagged
   runtime and constrained-generation claims from the headline.

## Three Biggest Gaps Versus The PRD

### Gap 1: Exact Verifiers Are Clean But Too Small

PRD FR-12 calls for verifiable reasoning with auditable provenance. `.467` showed one small KAN/MILP
proof, but Carnot still lacks scale curves, multiple constraint families, prefix-probability bounds,
prompt-to-code assurance, and graph-evidence energies. `.468` must diversify exact checker surfaces
while keeping every claim under a solver, executable constraint, or deterministic bound.

### Gap 2: Local SOTA Runtime Is Not Yet A Trustworthy Experimental Substrate

The mandated SOTA GGUF files are available, but the prior endpoint/cache artifacts were either blocked
or adversarially flagged. LLM-backed experiments must include the required local SOTA model specs, but
the milestone should only use live local inference as a substrate after an endpoint artifact records
long-enough execution, server command, logs, completion proof, and logprob/top-logprob proof.

### Gap 3: Continuous Self-Learning Has Governance But No Productive Update Path

PRD FR-11 needs autonomous directed self-learning. Carnot has repeatedly shown that blind memory, replay,
or promote-on-dev strategies are unsafe or inert. The next step is a SEVerA-style Search-Verify-Learn
loop: candidate memories or skills are generated, checked against formal contracts, and promoted only
when exact held-out and non-forgetting guards pass.

## Fresh Research Folded In

Added to `research-references.md` before this plan under `V468-PLANNER-REFERENCES`:

- `arXiv:2512.05439` and `uiuc-focal-lab/Beaver` - deterministic prefix-probability bounds for LLM
  constraints.
- `arXiv:2606.30247` - evidence-relative graph grounding with path energies and support regions.
- `arXiv:2603.25111` - SEVerA formally guarded self-evolving agents.
- `arXiv:2601.21048` - TACO test-time adaptation for unsupervised combinatorial optimization.
- `arXiv:2602.16665` - PLANCK direct p-spin/HUBO optimization without quadratization-first thinking.
- `arXiv:2603.01150` - implicitly parallel neuromorphic CSP solver design and partition telemetry.
- `arXiv:2606.00722`, `arXiv:2602.00612`, `arXiv:2508.10111`, and `eth-sri/constrained-diffusion` -
  CFG-constrained diffusion LMs, prefix reachability, and completion-existence checks.
- `arXiv:2405.21047` - grammar-aligned decoding caveat about distribution distortion.
- OpenReview `O3Kg4dLdpg` - code-based assurance of prompt-defined constraints.
- `arXiv:2602.02888` - HALT logprob time-series hallucination assessment, held for after substrate
  cleanup.
- `arXiv:2604.17091` and `arXiv:2602.07755` - GenericAgent/ALMA memory design pressure.
- `arXiv:2511.00907` and `arXiv:2505.11081` - EBT/ARM-EBM citation-lineage architecture signals.
- `guidance-ai/llguidance` - practical grammar-constrained decoding baseline.
- Extropic XTR-0/TSU and Logical Intelligence Kona/Aleph updates.

## Architecture For .468

```text
                           research-references.md
                                    |
                                    v
                 +--------------------------------------+
                 | Exp5095 archive .467 / activate .468 |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 | Exp5096 SOTA source ingestion audit  |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 | Exp5097 clean GGUF endpoint/cache    |
                 +--------------------------------------+
                         |                         |
          clean logprobs |                         | optional live LLM substrate
                         v                         v
       +--------------------------------+   +---------------------------------+
       | Exp5099 BEAVER prefix bounds   |   | Exp5104 constrained audit       |
       +--------------------------------+   +---------------------------------+

       +--------------------------------+   +---------------------------------+
       | Exp5098 KAN/PWA/MILP scale     |   | Exp5100 code-assurance checks   |
       +--------------------------------+   +---------------------------------+
                         |                         |
                         v                         v
       +--------------------------------+   +---------------------------------+
       | Exp5101 graph evidence energy  |   | Exp5102 direct HUBO/p-spin      |
       +--------------------------------+   +---------------------------------+
                         |                         |
                         v                         v
       +--------------------------------+   +---------------------------------+
       | Exp5103 TACO adaptive CSP      |   | Exp5105 SEVerA FR-11 memory     |
       +--------------------------------+   +---------------------------------+

                 +--------------------------------------+
                 | Exp5106 hardware partition telemetry |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 | Exp5107 capstone decision            |
                 +--------------------------------------+
```

## Phases

### Phase 0: Transition And SOTA Ingestion

Experiments: `exp5095`, `exp5096`

Archive the `.467` truth record, activate `.468`, and verify that the new source set is present and
mapped to experiments. `exp5096` is the formal SOTA-ingestion slot required by the local research
program. It must confirm the BEAVER, SEVerA, graph-evidence, TACO, p-spin, constrained-generation,
hardware, and Logical/Extropic source hooks before the milestone claims novelty.

### Phase 1: Runtime Provenance And Exact-Verifier Scale-Up

Experiments: `exp5097`, `exp5098`, `exp5099`, `exp5100`

`exp5097` repairs runtime truth by producing a clean endpoint/cache artifact or an actionable blocker.
It is not allowed to headline unless duration, server logs, completion proof, and logprob/top-logprob
proof pass adversarial review. In parallel, `exp5098` scales the clean KAN/PWA/MILP path, `exp5099`
prototypes BEAVER-style deterministic prefix bounds, and `exp5100` tests prompt-defined constraints as
executable code assurance.

### Phase 2: Evidence Energy, High-Order Constraints, And Adaptive Solver Help

Experiments: `exp5101`, `exp5102`, `exp5103`, `exp5104`

This phase expands the exact-verifier pivot into four adjacent mechanisms:

- Evidence-relative graph energy distinguishes contradictions from unsupported but possible claims.
- Direct HUBO/p-spin encoding tests whether high-order constraints avoid QUBO gadget blowup.
- TACO-style adaptation tests neural/energy warm starts with exact solver fallback.
- Constrained-decoding audit revisits `.467` STATIC CSR only with semantic no-op, distribution-distortion,
  and external grammar-baseline controls.

### Phase 3: Formal Self-Learning, Hardware Continuity, And Capstone

Experiments: `exp5105`, `exp5106`, `exp5107`

`exp5105` is the required continuous self-learning task. It uses a SEVerA-style Search-Verify-Learn
contract for memory/SOP updates and promotes nothing unless held-out and non-forgetting guards are
positive. `exp5106` keeps the hardware line alive with KV260 SSH/UIO-safe transcripts, GateMate detect
triage, PolarFire precheck, and partition/update telemetry. `exp5107` is deliberately ungated and must
aggregate clean wins, blockers, and flagged claims into one honest milestone decision.

## Dependency Graph

```text
exp5095
  -> exp5096
      -> exp5097
      -> exp5098
      -> exp5099
      -> exp5100
      -> exp5101
      -> exp5102
      -> exp5103
      -> exp5104
      -> exp5105
      -> exp5106

exp5095, exp5096, exp5097, exp5098, exp5099, exp5100,
exp5101, exp5102, exp5103, exp5104, exp5105, exp5106
  -> exp5107
```

Structured `gated_on` entries are used only where the conductor can skip wasted calls safely:

- `exp5099` waits for `exp5096.references_section_found == true`.
- `exp5104` waits for `exp5096.references_section_found == true`.
- No task depends structurally on `exp5097` being clean because exact-verifier diagnostics should still
  run with deterministic toy distributions or non-LLM checkers when local runtime remains blocked.

The capstone has no structured gate. It must run even if runtime or hardware tasks block.

## Hardware Requirements

- **Dual RTX 3090 / CUDA host:** required for `exp5097` and any task that invokes local SOTA inference.
  Use `scripts/experiment_template.py::cached_sota_pair()` patterns and resolved local `.gguf` files;
  never call `AutoTokenizer.from_pretrained()` on `-GGUF` repo IDs.
- **Mandated local GGUFs for every LLM-backed experiment:**
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **KV260:** SSH-only checks via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; use only the
  board SSH path and do not inspect host block devices. Any UIO/register interaction must be
  transcript-backed and safe.
- **GateMate A1 / DirtyJTAG:** detect/terminal-state triage only unless USB ID, IDCODE, and bitstream
  path are proven in the artifact.
- **PolarFire:** SSH reachability and hash-verified dispatch/precheck only; no flash/timing claim without
  transcript.
- **Extropic/TSU:** architecture and simulation reference only; no local TSU hardware target exists.
- **CPU-only exact solver fallback:** every exact-verifier task must have a deterministic CPU path so the
  milestone still produces useful science if GPU runtime is unavailable.

## Falsifiable Milestone Gates

- A headline exact-verifier win must come from an artifact with `flagged_adversarial=false`,
  `inference_substrate` matching the actual substrate, and an objective checker/bound/solver as the
  correctness authority.
- A live local SOTA claim must include one of the mandated GGUF models, server command/logs, duration,
  completion proof, and logprob/top-logprob proof.
- A constrained-generation claim must report semantic controls and cannot headline on syntactic validity
  alone.
- A self-learning claim must report held-out and non-forgetting deltas and must refuse promotion unless
  both are positive under exact guards.
- A hardware claim may report reachability, transcript, and telemetry. It may not report acceleration
  unless board execution and timing are directly measured.

## Expected Deliverables

- `results/experiment_5095_archive_467_activate_468.json`
- `results/experiment_5096_sota_ingestion_v468.json`
- `results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json`
- `results/experiment_5098_kan_pwa_milp_scale_v2.json`
- `results/experiment_5099_beaver_prefix_bound_verifier_v468.json`
- `results/experiment_5100_constrainprompt_code_assurance_v468.json`
- `results/experiment_5101_incomplete_graph_evidence_energy_v468.json`
- `results/experiment_5102_hubo_pspin_direct_energy_v468.json`
- `results/experiment_5103_taco_adaptive_csp_heuristic_v468.json`
- `results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json`
- `results/experiment_5105_fr11_severa_guarded_memory_v468.json`
- `results/experiment_5106_hardware_partition_telemetry_v468.json`
- `results/experiment_5107_capstone_v468.json`
