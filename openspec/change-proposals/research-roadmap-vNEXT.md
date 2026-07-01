# Research Roadmap vNEXT: 2026.07.470

## Milestone Title

POST-FOVER STRUCTURED ENERGY + AUDITABLE FR-11 + SAMPLER/CERTIFICATE SCALE

## Why This Milestone Exists

Milestone `2026.07.469` completed all active tasks, but it did not produce a clean
verifier-moat headline. The important result is not "try FoVer again"; it is the opposite:
the FoVer in-domain selector premise was retracted because the candidate-selection pool was
not well posed. `exp5121` explicitly recommended retiring same-verdict FoVer selector,
audit, and FoVer-residual FR-11 reruns until a genuinely different multi-candidate corpus
or benchmark exists.

What `.469` proved:

1. **FoVer selector path is blocked and should not be carried forward.** `exp5111` found
   that FoVer's real classification task is better framed as learned discriminator versus
   cheap non-learned baseline, not verifier selection versus self-consistency. The cheap
   baseline AUROC was `0.9635`, verifier AUROC was `0.9663`, and the CI included zero.
   `exp5112`, `exp5113`, `exp5115`, and `exp5118` were consequently gate-blocked or
   skipped. This is a retirement signal for that path, not an invitation to rerun it.
2. **KAN post-wall verification is a clean positive.** `exp5108` measured the exact-MILP
   wall at `N=10`; `exp5114` changed technique and produced conservative post-wall
   progress at `N=100` with false-property and near-margin controls intact.
3. **Solver/sampling references are clean enough to scale.** `exp5116` built an exact-
   checked CPU HUBO/2D-PT reference. `exp5117` preserved exact labels and shipped a TACO
   harm gate. These are stronger forward substrates than the blocked FoVer transfer.
4. **FR-11 remains safe but nonproductive.** The planned FoVer residual-memory task did not
   run because the upstream selector path was invalid. The next FR-11 experiment must use
   a different auditable stream and must keep no-promotion as a valid outcome.
5. **Local SOTA runtime is useful but still quarantined.** `exp5119` found cached mandated
   GGUFs and recorded completion/logprob evidence, but adversarial verification flagged the
   runtime artifact. Any LLM-backed claim in `.470` must be gated behind a cleaner endpoint,
   cache, and duration receipt.
6. **Hardware continuity is ready for authenticated timing attempts, not speed claims.**
   `exp5120` produced clean CPU-reference residual telemetry and showed KV260/PolarFire
   reachability, while preserving `no_speedup_claim=true`.

## Three Biggest Gaps Versus The PRD

### Gap 1: Verifier Utility Needs A Well-Posed Structured Reasoning Target

PRD FR-12 is not satisfied by high AUROC on a confounded classifier or by reranking where no
multi-candidate headroom exists. Carnot needs a structured benchmark where deterministic
constraints are the ground truth, candidates have genuine oracle headroom, and a learned
energy can be tested against cheap baselines without label leakage.

### Gap 2: Continuous Self-Learning Still Has No Safe Nonzero Promotion Path

PRD FR-11 requires autonomous directed self-learning. The current loop has strong rollback
and nonforgetting discipline, but repeated no-promote outcomes show that raw replay memory
and FoVer residuals are not sufficient. Recent deployment-time learning work points to a
better next step: nonparametric case selection over exact-solver traces, with bandit
telemetry and no-weight updates.

### Gap 3: Exact Verification Must Become Explainable And Hardware-Compatible

The clean KAN and HUBO results are useful, but they are still research artifacts. To move
toward the Phase 3 foundation-model architecture, certificates need faithful explanations,
solver/sampler telemetry needs adaptive temperature and residual-energy evidence, and board
work must move from continuity matrices to authenticated command transcripts.

## Fresh Research Folded In

Added to `research-references.md` before this plan under `V470-PLANNER-REFERENCES`:

- `arXiv:2605.18871` Distributional EBMs for structured LLM reasoning: learned quality
  scorer plus deterministic constraint penalties and uncertainty-triggered regeneration.
- `arXiv:2605.28020` Energy-Based Decoding: training-free reward-tilted decoding for
  frozen LLMs, usable only after local GGUF runtime is clean.
- `arXiv:2603.12248` Energy-Based Fine-Tuning: sequence-level feature matching as a
  medium-term energy-training signal, not a `.470` reproduction.
- `arXiv:2606.29713` SEVA: self-evolving verifier with process rewards, plus the warning
  that self-evolution can create benchmark specialists rather than generalists.
- `arXiv:2605.06702` CASCADE and `arXiv:2601.18510` JitRL: deployment-time learning via
  episodic memory and no-weight action/logit modulation.
- `arXiv:2603.10060` tool receipts: signed command/tool evidence for practical agent
  hallucination detection.
- `arXiv:2606.24414` cycle-consistent explanation of formal-verification certificates.
- `arXiv:2510.17376` AdapTrack and constrained-decoding GitHub survey: backtracking and
  semantic reachability as antidotes to syntax-only constrained decoding.
- `arXiv:2601.13542` and `arXiv:2603.09251`: adaptive replica-exchange temperatures and
  reversibility-constrained sampling for the HUBO/2D-PT path.
- Extropic TSU and Logical Intelligence Kona/Aleph remain architecture pressure only; no
  local TSU/Kona execution claim is planned.

## Architecture For .470

```text
                  research-references.md V470 block
                                |
                                v
           +-----------------------------------------+
           | exp5122 archive .469 -> activate .470  |
           +-----------------------------------------+
                                |
                                v
           +-----------------------------------------+
           | exp5123 source/failure/scope audit      |
           +-----------------------------------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
 +-------------------------------+   +-------------------------------+
 | exp5124 clean SOTA runtime    |   | exp5128 KAN cert explanation  |
 | GGUF completion/logprob gate  |   +-------------------------------+
 +-------------------------------+                  |
              |                                     v
              v                       +-------------------------------+
 +-------------------------------+     | exp5129 HUBO adaptive 2D-PT  |
 | exp5125 structured pool       |     +-------------------------------+
 | exact constraints + SOTA cands|                   |
 +-------------------------------+                  v
              |                       +-------------------------------+
              v                       | exp5130 TACO held-out scale   |
 +-------------------------------+     +-------------------------------+
 | exp5126 distributional EBM    |                   |
 | ranker/abstainer             |                   v
 +-------------------------------+     +-------------------------------+
              |                       | exp5131 FR-11 case policy     |
              v                       | CASCADE/JitRL, no weights     |
 +-------------------------------+     +-------------------------------+
 | exp5127 adversarial audit +   |                   |
 | tool-receipt provenance       |                   v
 +-------------------------------+     +-------------------------------+
                                      | exp5132 hardware transcripts  |
                                      +-------------------------------+
                                                   |
                                                   v
                              +---------------------------------------+
                              | exp5133 capstone .470 decisions       |
                              +---------------------------------------+
```

The LLM-heavy branch (`exp5125`-`exp5127`) is structurally gated behind
`exp5124.sota_runtime_clean == true`. If the endpoint remains dirty, the conductor should
skip those tasks quickly while KAN, sampler, FR-11, and hardware work continue.

## Phases

### Phase 0: Transition, Source Freshness, And Scope Discipline

Experiments: `exp5122`, `exp5123`

Archive `.469`, preserve the true close-state, and preflight `.470` for two planning
hazards: accidental FoVer selector reruns and LLM tasks without mandated SOTA GGUF
`MODEL_SPECS`. `exp5123` also verifies that the V470 reference block exists and maps each
fresh source either to a concrete task or to architecture-only context.

### Phase A: Clean Runtime And Structured Energy Utility

Experiments: `exp5124`, `exp5125`, `exp5126`, `exp5127`

`exp5124` repairs the local SOTA GGUF endpoint evidence floor with real live-duration,
cache, completion, and logprob receipts. Downstream LLM tasks run only if that artifact is
clean. `exp5125` builds a non-FoVer structured reasoning candidate pool with deterministic
constraints, oracle headroom, and mandated SOTA local GGUF candidates. `exp5126` evaluates a
Distributional-EBM-style ranker/abstainer against cheap baselines, with deterministic
constraint penalties as ground truth. `exp5127` audits any positive ranker result using
tool/file receipts, leakage checks, duplicate checks, and baseline adequacy.

### Phase B: Exact Certificates, Sampler Adaptation, And Solver Help

Experiments: `exp5128`, `exp5129`, `exp5130`

`exp5128` extends the KAN post-wall result across independent property families and adds a
cycle-consistent certificate explanation audit. `exp5129` upgrades the exact-checked HUBO
2D-PT reference with online temperature-ladder adaptation and reversibility/detailed-balance
telemetry. `exp5130` scales the TACO harm-gated exact-solver helper on held-out CSP families
and records whether the adaptive sampler helps or harms solver effort.

### Phase C: Continuous Self-Learning And Hardware Continuity

Experiments: `exp5131`, `exp5132`

`exp5131` is the required continuous self-learning task. It replaces the blocked FoVer
residual route with a CASCADE/JitRL-style nonparametric case policy over exact-solver traces:
select cases, estimate action/heuristic advantages, apply only no-weight updates, and roll
back unless held-out utility and nonforgetting are both safe. `exp5132` attempts authenticated
board timing or records honest blockers for KV260, GateMate, and PolarFire while preserving
residual-energy telemetry and no TSU execution claim.

### Phase Z: Capstone

Experiment: `exp5133`

Aggregate the runtime branch, structured-energy utility, KAN/certificate breadth, sampler
and TACO scale, FR-11 promotion safety, and hardware status. The capstone is intentionally
ungated: missing or flagged artifacts become explicit gaps, not zeros for unrelated axes.

## Dependency Graph

```text
exp5122
  -> exp5123
      -> exp5124
          -> exp5125
              -> exp5126
                  -> exp5127
      -> exp5128
      -> exp5129
          -> exp5130
              -> exp5131
      -> exp5132

exp5122, exp5123, exp5124, exp5125, exp5126, exp5127,
exp5128, exp5129, exp5130, exp5131, exp5132
  -> exp5133
```

Structured `gated_on` entries used in `research-roadmap-next.yaml`:

- `exp5125` waits for `exp5124.sota_runtime_clean == true`.
- `exp5126` waits for `exp5125.structured_pool_ready == true`.
- `exp5127` waits for `exp5126.distributional_energy_delta > 0.0`.
- `exp5130` waits for `exp5129.adaptive_2dpt_ready == true`.
- `exp5131` waits for `exp5130.heldout_csp_trace_suite_ready == true`.

The capstone is deliberately ungated.

## Hardware Requirements

- **Dual RTX 3090 / CUDA host:** required for `exp5124` and any downstream local
  SOTA-GGUF generation. Use llama.cpp and `cached_sota_pair()` from
  `scripts/experiment_template.py`; never call Hugging Face `AutoTokenizer` on `-GGUF`
  repositories.
- **Mandated local GGUFs for every LLM-backed experiment:**
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **CPU exact-solver path:** required for `exp5128`-`exp5131`; GPU is optional unless a
  task explicitly records SOTA LLM generation.
- **KV260:** SSH-only checks through commands such as
  `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Do not inspect host block
  devices. UIO/register interaction must include command transcript, workload hash,
  timing, and sample-quality evidence.
- **GateMate A1 / DirtyJTAG:** detection and flash attempts only if USB ID, IDCODE, and
  toolchain commands are transcript-backed.
- **PolarFire:** SSH/precheck and dispatch only with hash-matched workload and timing
  transcript.
- **Extropic TSU / XTR-0:** architecture/simulation context only; no local execution target
  exists.

## Planned Deliverables

| Task | Deliverable |
|---|---|
| `exp5122` | `results/experiment_5122_archive_469_activate_470.json` |
| `exp5123` | `results/experiment_5123_v470_source_scope_audit.json` |
| `exp5124` | `results/experiment_5124_clean_sota_runtime_provenance_v470.json` |
| `exp5125` | `results/experiment_5125_structured_reasoning_pool_v470.json` |
| `exp5126` | `results/experiment_5126_distributional_energy_ranker_v470.json` |
| `exp5127` | `results/experiment_5127_structured_energy_adversarial_audit_v470.json` |
| `exp5128` | `results/experiment_5128_kan_certificate_explanation_v470.json` |
| `exp5129` | `results/experiment_5129_hubo_adaptive_2dpt_v470.json` |
| `exp5130` | `results/experiment_5130_taco_sampler_heldout_scale_v470.json` |
| `exp5131` | `results/experiment_5131_fr11_case_policy_self_learning_v470.json` |
| `exp5132` | `results/experiment_5132_authenticated_board_timing_v470.json` |
| `exp5133` | `results/experiment_5133_capstone_v470.json` |

## Success Criteria

1. No FoVer selector, FoVer audit, or FoVer residual-memory rerun is proposed unless the
   task explicitly changes benchmark and includes a prior-failure retirement clause.
2. Every LLM-backed task names at least one mandated SOTA local GGUF in `MODEL_SPECS` and
   is gated behind clean runtime provenance.
3. A structured-energy result headlines only if deterministic constraints, cheap baselines,
   no-leak controls, and adversarial audit all pass.
4. KAN certificate work reports breadth and explanation faithfulness, not just another
   toy exact-MILP timing number.
5. HUBO/TACO tasks preserve exact correctness labels and report harmful-instance counts.
6. FR-11 promotion happens only if held-out utility improves and nonforgetting holds;
   no-promotion remains a complete, honest outcome.
7. Hardware artifacts make no speedup or TSU claim without authenticated command
   transcripts, workload hashes, timings, and sample-quality evidence.
