# Research Roadmap vNEXT: V477 -- Verifier-Set Hardening, ARC Search-Frontier Gates, Self-Learning Memory, Hardware Continuity

**Milestone:** `2026.07.477`
**Status:** Planner-authored after completion of all `2026.07.476` tasks.
**Prepared:** 2026-07-03
**Predecessor:** `2026.07.476`
**Execution manifest:** `research-roadmap-next.yaml`

## What `.476` proved

`.476` was a recovery milestone after `.475`'s poison-test cascade. Its main value was not a new ARC
level or a new headline verifier; it separated usable signals from exhausted threads.

| Area | `.476` result | Planning implication for `.477` |
|---|---|---|
| GAP-1 transpose/orientation | `exp5205` found a compact AutoPyVerifier-style set-search positive: pass@2 `0.221757` vs always-on `0.087866` and the refuted single directional-adjacency baseline `0.150628`; it captured `47/239` transpose misvotes. | Harden the result across splits, then promote only if the held-out gate remains positive. |
| GAP-4 same-shape rule execution | `exp5197` reached only `n=62` because the source pool was exhausted; signal remained `4/0` discordant wins/losses, `p=0.125`, below the >=6-win significance floor. Local SOTA generator arm used `unsloth/Qwen3.6-35B-A3B-GGUF` but mostly produced `no_code`. | Generate a new feasible candidate pool with SOTA local GGUFs plus repair guards before another scale test. |
| GAP-4891 ARC trajectory enumeration | `exp5198` MAP/landmark prestage banked zero levels over pruner-only; no source read, no offline BFS, but the enumeration wall persisted. | Stop adding goal detectors. Test whether frontier continuity / landmark decomposition changes what reaches the frontier. |
| Hidden-state verifier | `exp5200` trained probe accuracy was `0.100`, tuned SC `0.075`, but CLUE and RCS were also `0.100`; final-layer-only extraction did not beat zero-training controls. | One last v3 should test intermediate/chunk/halting signals. If null again, retire this MMLU-Pro hidden-state path. |
| DiffusionGemma | `exp5196` exhausted vLLM/HF loading and retired the loading thread for now. | Do not propose another DiffusionGemma loader task until upstream loader status materially changes. |
| Hardware | `exp5201` confirmed KV260 and PolarFire reachable, no speedup claim; GateMate remained blocked at DirtyJTAG/JTAG physical-protocol level. | Keep one hardware continuity slot, SSH-only for KV260, hash/correctness only. |
| Governance | `exp5203` produced remediation options for dishonest verifier names; `exp5204` fixed the exclusion-manifest lint bug class; `exp5206` reconciled architecture and retired threads. | Apply the low-risk verifier-authenticity remediation; keep conductor untouched. |

## Three biggest gaps to PRD vision

1. **Verifier positives are not yet production verifiers.** The PRD's verifiable reasoning vision needs
   registry-grade constraints with held-out evidence, not one-off positive artifacts. GAP-1 now has a
   promising set-search signal; GAP-4 still has a source-pool bottleneck. `.477` turns these into
   gated promotion/retirement decisions.

2. **The ARC live agent still fails at trajectory enumeration.** GAP-4891 repeatedly shows that Carnot can
   identify promising states but cannot reliably enumerate paths to them. `.477` therefore tests
   search-frontier mechanisms: PAW amortization feasibility first, then frontier continuity/landmark
   decomposition against the public-game live-runtime discipline.

3. **Continuous self-learning is not yet a durable capability.** Prior self-learning work is scattered
   across probes and retros. `.477` creates a small verifier-memory promotion loop: failures create
   candidate predicates/rules, deterministic checks validate them, held-out gates promote them, and nulls
   roll back.

## Current architecture

```text
                         research-references.md
                 (SOTA ingestion: FALCON, STV, ADVENT, PAW,
                  EBT citation trail, thermodynamic hardware)
                                      |
                                      v
         +-------------------- Carnot verifier stack --------------------+
         |                                                               |
         |  GAP-1 set-search        GAP-4 rule execution                 |
         |  exp5209 -> exp5210      exp5211 -> exp5212                  |
         |      |                       |                                |
         |      +-----------+-----------+                                |
         |                  v                                            |
         |        exp5214 self-learning verifier memory                  |
         |        (promote/rollback predicate sets)                      |
         |                                                               |
         |  Hidden-state/internal verifier v3                            |
         |  exp5213: layer/chunk/halting sweep or retire path            |
         +------------------+--------------------------------------------+
                            |
                            v
         +---------------- ARC live/search path ----------------+
         | exp5215 PAW compile-amortization gate                |
         | exp5216 frontier continuity + landmark decomposition |
         | registry precheck, no source reading, reproduction   |
         | gate before any level claim                          |
         +------------------+-----------------------------------+
                            |
                            v
         +---------------- ops / hardware / governance ---------+
         | exp5217 KV260/PolarFire/GateMate continuity          |
         | exp5218 verifier-authenticity remediation            |
         | exp5219 capstone reconciliation                      |
         +------------------------------------------------------+
```

## SOTA findings incorporated

- **FALCON (`arXiv:2602.01090`)**: grammar-constrained decoding + semantic repair + adaptive Best-of-N.
  Used as the guardrail pattern for local SOTA verifier/rule candidate generation.
- **Distributional EBM structured verification (`arXiv:2605.18871`)**: useful for uncertainty and
  abstention around structured artifacts, not for reopening retired external generated-text scorers.
- **Energy-Based Decoding (`arXiv:2605.28020`)**: optional decoding arm for frozen local models if the
  loader exposes token-level steering; any claim must be measured by pass@2/discordant gates.
- **STV, DeepVerifier, ADVENT (`arXiv:2605.30290`, `2601.15808`, `2607.01585`)**: support the self-learning
  verifier-memory design: failure taxonomy, predicate invention, mechanical verification, knowledge-pool
  promotion.
- **PAW (`arXiv:2607.02512`)**: first use is an amortization/data gate, not mid-episode LoRA infra.
- **EBT citation trail (`2507.02092` -> FPRM/LoopUS/CEM)**: motivates hidden-state v3 to test iterative
  latent/chunk/halting signals rather than another final-layer-only probe.
- **GRS-KAN/KANFIS/KAF**: noted for later interpretable differentiable constraint energies, not selected
  as a `.477` build target.
- **Extropic / p-computers / thermodynamic AI**: confirms the long-run hardware thesis while keeping
  `.477` hardware work to correctness/reachability smokes.
- **Logical Intelligence public posts**: reinforce the product stance "EBRM/verifier as correctness
  layer, LLM as interface," but do not provide reproducible baselines to copy.

## Phase design

### Phase 0 -- Transition and SOTA ingestion

`exp5207` archives `.476` and activates `.477`. `exp5208` is a reserved SOTA-ingestion update that rechecks
the freshest papers/repos and the Semantic Scholar ARM-EBM citation trail that rate-limited during this
planning pass. This satisfies the standing SOTA-ingestion rule and prevents stale planning.

### Phase A -- Verifier gaps and self-learning

`exp5209` hardens the `exp5205` GAP-1 set-search positive. `exp5210` is mechanically gated on
`exp5209.gap1_hardened_positive == true` and promotes the verifier only after the held-out gate clears.

`exp5211` uses mandated SOTA local GGUF models to build a larger GAP-4 feasible candidate pool with
FALCON-style syntax/semantic repair. `exp5212` is gated on `candidate_pool_n >= 120` and
`gap4_expansion_usable == true` before spending the scale-validation budget.

`exp5213` runs hidden-state verifier v3. It must include the SOTA GGUF model specs and either find an
intermediate/chunk/halting signal that beats SC and all controls, or recommend retiring the MMLU-Pro
hidden-state path.

`exp5214` is the mandatory continuous self-learning experiment. It builds a small verifier-memory
promotion/rollback ledger using GAP-1/GAP-4 evidence and ADVENT/STV-style failure classes.

### Phase B -- ARC search-frontier gates

`exp5215` applies PAW only as an amortization gate: measure remaining episode length and plausible compile
cost before building any mid-episode compile infrastructure.

`exp5216` attacks GAP-4891 directly with frontier continuity and landmark decomposition. It is not another
goal detector. It must registry-precheck, avoid duplicate solves, avoid source reading/offline BFS, and
declare `solve_provenance` for any banked level.

### Phase C -- Hardware, authenticity, capstone

`exp5217` keeps hardware continuity alive: KV260 via SSH-only precondition, PolarFire hash smoke, GateMate
JTAG/physical-protocol follow-up. No latency/speedup claim.

`exp5218` applies the low-risk verifier-authenticity remediation from `exp5203` so misleading verifier
names cannot remain headline-eligible without explicit warnings/tests.

`exp5219` reconciles the milestone results across `openspec/`, `_bmad/traceability.md`, `ops/status.md`,
`ops/changelog.md`, and the relevant gap/registry docs.

## Dependency graph

```text
exp5207 archive/activate
  -> exp5208 SOTA ingestion
  -> exp5209 GAP-1 held-out hardening
       -> exp5210 GAP-1 registry promotion [gate: gap1_hardened_positive == true]
  -> exp5211 GAP-4 SOTA local candidate expansion
       -> exp5212 GAP-4 scale validation [gates: candidate_pool_n >= 120,
                                           gap4_expansion_usable == true]
  -> exp5213 hidden-state verifier v3
  -> exp5214 continuous self-learning verifier memory
  -> exp5215 ARC PAW amortization gate
       -> exp5216 ARC frontier continuity / landmark decomposition
  -> exp5217 hardware continuity
  -> exp5218 verifier-authenticity remediation
       -> exp5219 capstone reconciliation
```

The graph is deliberately shallow. GAP-1 promotion and GAP-4 validation are gated because they would waste
runtime if their prerequisites fail. Self-learning can still run from existing `.476` evidence even if a
new gate fails.

## Hardware and model requirements

- **Primary compute:** local dual RTX 3090 host. Any LLM experiment must use at least one mandated SOTA
  GGUF in `MODEL_SPECS`, preferably via `cached_sota_pair()`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **Fallback models:** legacy tiny models are allowed only as smoke tests when SOTA GGUF cache resolution
  fails, and artifacts must mark expected output quality as poor.
- **KV260:** precondition is exactly SSH reachability via
  `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Host `/dev/mmcblk*` checks are forbidden.
- **PolarFire:** SSH reachability plus workload hash smoke only.
- **GateMate:** continue only diagnostic narrowing. No same-result "detect only" task; report whether the
  block is USB, JTAG protocol, permissions, firmware/toolchain, cable/port, or physical board state.
- **No hardware speedup claim** may appear in `.477` unless a task produces an end-to-end sampler workload
  with correctness-preserving baseline comparison. No such task is planned.

## Retired-scope safeguards

- Do not reopen DiffusionGemma loading until upstream loader status changes materially.
- Do not propose Phase-D external generated-text/logprob scorer work; hidden-state/internal-representation
  work remains permitted only because it scores internal signals.
- Do not rerun the retired generation-axis first-contact exploration-signal class. `exp5216` is scoped to
  frontier continuity/landmark decomposition and must document how it differs.
- Do not use host SD-card preconditions for KV260.
- Do not claim ARC solves from outer-loop RE, source reading, exhaustive offline BFS, or unreachable
  dev-only adapters.

## Exit criteria

The milestone succeeds if it leaves at least one of these in a cleaner state:

1. GAP-1 set-search is either promoted to registry-grade evidence or rejected by a held-out hardening gate.
2. GAP-4 has a real expanded feasible candidate pool and a scale-validation result, or the local SOTA
   generator path is retired with a concrete failure mode.
3. Hidden-state verifier v3 either beats all controls on a non-final-layer/chunk signal or retires the
   MMLU-Pro hidden-state path.
4. ARC PAW/frontier tasks produce a falsifiable gate result without duplicate/off-path solve claims.
5. Continuous self-learning gains a durable verifier-memory artifact with rollback semantics.

The capstone must report nulls plainly and must not headline any artifact flagged adversarial or outside
the live-path/provenance discipline.
