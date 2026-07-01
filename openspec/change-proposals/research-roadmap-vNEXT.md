# Research Roadmap vNEXT: 2026.07.469

## Milestone Title

FOVER IN-DOMAIN SELECTION + POST-WALL KAN ABSTRACTION + CONTINUITY REPAIR

## Why This Milestone Exists

Milestone `2026.07.468` completed with useful exact-verifier evidence, but not with a clean local runtime
substrate, not with a productive FR-11 update, and not with a verifier-moat result on an oracle-distinct
domain. A post-milestone operator-directed artifact, `exp5108`, also answered the KAN exact-MILP scale
question: the current exact PWA/MILP encoding solved N=10 in 120.9s, timed out at N=20, and is far below
the realistic N=100 reference. The next milestone should therefore move the center of gravity to the
FoVer in-domain redirect, while turning exact-verifier scale work toward different mechanisms rather than
more toy exact-MILP wins.

What `.468` and the immediate follow-up proved:

1. **Bounded exact-verifier work remains the cleanest positive path.** `exp5098`, `exp5101`, `exp5102`,
   and `exp5103` produced clean bounded positives for KAN/PWA/MILP, graph evidence energy, direct
   HUBO/p-spin encoding, and TACO-style exact-solver help.
2. **The KAN exact-MILP scale wall is now measured.** `exp5108` reached N=10, timed out at N=20, preserved
   adversarial controls where solved, and did not approach the realistic N=100 KAEM reference. Repeating
   the same exact-MILP sweep is no longer research progress.
3. **The FoVer in-domain redirect remains unrun and is now the highest-value verifier-moat question.**
   Prior cross-domain FoVer selector attempts (`exp4305`, `exp4314`) found real headroom but collapsed
   under label ablation. The untested question is in-domain FoVer selection versus tuned
   self-consistency on a proper n>=150 oracle-distinct pool.
4. **Local SOTA runtime is still blocked.** `exp5097` resolved mandatory GGUF files but could not prove
   live completion/logprob readiness. LLM-backed claims remain downstream of a clean endpoint artifact.
5. **Constrained-generation evidence was not clean enough to keep as a headline.** `exp5099`, `exp5100`,
   and `exp5104` were useful diagnostics but adversarially flagged or semantic-audit-only.
6. **FR-11 self-learning is safe but inert.** `exp5105` preserved contracts and did not promote unsafe
   memory; held-out utility stayed at zero and the artifact was not a clean promotion result.
7. **Hardware progress is continuity, not acceleration.** `exp5106` kept KV260, GateMate, and PolarFire
   context alive without a speedup claim. The next hardware task should measure residual-energy
   methodology or record an honest blocker, not infer acceleration from static mappings.

## Three Biggest Gaps Versus The PRD

### Gap 1: The Verifier Moat Has Not Been Tested Where Headroom Exists

PRD FR-12 calls for verifiable reasoning that improves outcomes, not only discriminates errors after the
fact. MuSR/math/CSP selector attempts have repeatedly lacked oracle-distinct headroom. FoVer is the
known oracle-distinct domain with headroom (`oracle@K` around 0.77 versus vote around 0.27), but the only
previous FoVer selector result was cross-domain and label-ablation-collapsed. `.469` must answer the
in-domain FoVer selection question directly.

### Gap 2: Exact Verification Needs Alternatives Beyond Exact-MILP Scaling

The KAN exact-MILP scale wall means Carnot needs post-wall formal methods: abstraction refinement,
piece-budget allocation, decomposition, conservative sampling bounds, and high-order energy methods that
keep exact checks on small instances. This milestone should avoid another N<=3 proof and instead test
whether a different KAN abstraction can produce useful conservative certificates beyond `exp5108`.

### Gap 3: Continuous Self-Learning Still Lacks A Productive, Auditable Update Source

PRD FR-11 requires autonomous directed self-learning. Carnot has governance and rollback, but no safe
nonzero update path. FoVer selection residuals are a better update stream than blind replay: they contain
formal step labels, candidate-selection failures, and auditable error families. `.469` should use FoVer
residuals to test contract-guarded memory/SOP promotion with held-out and non-forgetting guards.

## Fresh Research Folded In

Added to `research-references.md` before this plan under `V469-PLANNER-REFERENCES`:

- `arXiv:2505.15960` FoVer and its code/Hugging Face records - formal-tool labels for step-level
  verifier training and Best-of-K selection.
- `arXiv:2605.10141` FormalRewardBench - formal theorem-proving reward-model evaluation with
  error-injected controls.
- OpenReview `QJTSAvHFQn` - the self-verification cliff between candidate generation and selection.
- `arXiv:2602.06737` - KAN property verification through optimal piecewise-affine abstraction and
  abstraction-budgeting.
- `arXiv:2601.09037` - FPGA p-bit methods with sparsification and two-dimensional parallel tempering.
- `arXiv:2606.25313` - million p-bit probabilistic computer with residual-energy and partition telemetry.
- `arXiv:2602.01090` FALCON - hard feasibility for LLM-based combinatorial optimization through
  grammar-constrained decoding plus semantic repair and adaptive Best-of-N sampling.
- Verus-SpecGym / Verus-SpecBench - executable specification faithfulness checks using adversarial
  edge cases rather than LLM-judge-only evaluation.
- `arXiv:2512.16762` and `arXiv:2604.00555` - EBT/ARM-EBM citation-lineage architecture signals.
- `arXiv:2507.02092` and `arXiv:2512.15605` - primary EBT and ARM-as-EBM sources; architecture
  pressure only, not a `.469` training reproduction.
- Extropic XTR-0/TSU and Logical Intelligence Kona/Aleph updates - architecture context, not local
  execution claims.

## Architecture For .469

```text
                         research-references.md
                                  |
                                  v
              +-----------------------------------------+
              | exp5109 archive .468 + exp5108 baseline |
              +-----------------------------------------+
                                  |
                                  v
              +-----------------------------------------+
              | exp5110 V469 source ingestion audit     |
              +-----------------------------------------+
                                  |
           +----------------------+----------------------+
           |                                             |
           v                                             v
+----------------------------+              +------------------------------+
| exp5111 FoVer pool n>=150  |              | exp5119 GGUF endpoint RCA    |
+----------------------------+              +------------------------------+
           |                                             |
           v                                             v
+----------------------------+              +------------------------------+
| exp5112 in-domain selector |              | future live LLM experiments  |
+----------------------------+              +------------------------------+
           |
           +-----------------------------+
           |                             |
           v                             v
+----------------------------+  +------------------------------+
| exp5113 adversarial audit  |  | exp5118 FR-11 FoVer memory   |
+----------------------------+  +------------------------------+
           |
           v
+----------------------------+  +------------------------------+
| exp5115 graph/FoVer energy |  | exp5114 KAN abstraction      |
+----------------------------+  +------------------------------+
                                  |
                                  v
              +-----------------------------------------+
              | exp5116 HUBO 2D-PT CPU reference        |
              +-----------------------------------------+
                                  |
                                  v
              +-----------------------------------------+
              | exp5117 TACO harm-gated scale           |
              +-----------------------------------------+
                                  |
                                  v
              +-----------------------------------------+
              | exp5120 hardware residual telemetry     |
              +-----------------------------------------+
                                  |
                                  v
              +-----------------------------------------+
              | exp5121 capstone decision               |
              +-----------------------------------------+
```

## Phases

### Phase 0: Transition And Source Freshness

Experiments: `exp5109`, `exp5110`

Archive the `.468` close-state, include the post-milestone `exp5108` KAN scale-wall result, and activate
`.469` without modifying the active `research-roadmap.yaml`. Verify that the V469 source set is present,
that each source maps to a task or is explicitly background-only, and that the FoVer redirect is treated
as the primary science question.

### Phase 1: FoVer In-Domain Verifier Selection

Experiments: `exp5111`, `exp5112`, `exp5113`

Build the n>=150 FoVer candidate-selection pool, then run in-domain train/test verifier selection versus
tuned self-consistency. The selector must be oracle-distinct, must not use answer-key leakage, and must
survive the mandatory shuffled-label ablation. The audit is separate so a positive selector result cannot
headline until leakage, label ablation, vote tuning, CI handling, and Verus-SpecGym-style adversarial
edge-case controls are checked independently.

### Phase 2: Exact-Verifier Alternatives And Solver Help

Experiments: `exp5114`, `exp5115`, `exp5116`, `exp5117`

Move beyond the measured exact-MILP wall. `exp5114` tests KAN abstraction-refinement or decomposition
against `exp5108` with false-property and margin controls. `exp5115` transfers graph-evidence energy to
FoVer step/support traces. `exp5116` builds a CPU exact-checked HUBO/p-spin two-dimensional
parallel-tempering reference before any board claim. `exp5117` scales TACO-style exact-solver help and
learns a harm gate so `.468`'s helpful average does not hide harmful instances. FALCON is treated as a
constraint-generation lesson for semantic repair and adaptive sampling, not as a syntax-only headline.

### Phase 3: Continuous Self-Learning, Runtime Repair, And Hardware Continuity

Experiments: `exp5118`, `exp5119`, `exp5120`

`exp5118` is the required continuous self-learning task. It uses FoVer selector residuals to propose
memory/SOP updates and promotes only when exact contracts, held-out utility, and non-forgetting guards
pass. `exp5119` is runtime root-cause work for the local SOTA GGUF endpoint and logprob cache. `exp5120`
keeps hardware continuity honest with KV260 residual-energy methodology, GateMate/PolarFire prechecks,
and no speedup claim without authenticated board timing.

### Phase Z: Capstone

Experiment: `exp5121`

Aggregate all artifacts into one decision: FoVer moat status, selector-audit status, post-wall KAN path,
exact-solver alternatives, FR-11 promotion safety, runtime readiness, hardware status, and the next
milestone's focus. The capstone has no structured gate and must run even if runtime or hardware remains
blocked.

## Dependency Graph

```text
exp5109
  -> exp5110
      -> exp5111
          -> exp5112
              -> exp5113
              -> exp5118
          -> exp5115
      -> exp5114
      -> exp5116
      -> exp5117
      -> exp5119
      -> exp5120

exp5109, exp5110, exp5111, exp5112, exp5113, exp5114, exp5115,
exp5116, exp5117, exp5118, exp5119, exp5120
  -> exp5121
```

Structured `gated_on` entries are used where the conductor can skip work safely:

- `exp5111` waits for `exp5110.references_section_found == true`.
- `exp5112` waits for `exp5111.pool_n >= 150` and `exp5111.headroom_present == true`.
- `exp5113` waits for `exp5112.selection_result_available == true`.
- `exp5115` waits for `exp5111.pool_n >= 150`.
- `exp5118` waits for `exp5112.selection_result_available == true`.

The capstone is deliberately ungated.

## Hardware Requirements

- **Dual RTX 3090 / CUDA host:** required only for `exp5119` and any FoVer task that chooses to generate
  new LLM candidates instead of using cached candidates. Use `scripts/experiment_template.py` local GGUF
  cache helpers and llama.cpp; never call Hugging Face `AutoTokenizer` on `-GGUF` repositories.
- **Mandated local GGUFs for every LLM-backed experiment:**
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **KV260:** SSH-only checks through `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Do not
  inspect host block devices. Any UIO/register interaction must be transcript-backed and safe.
- **GateMate A1 / DirtyJTAG:** detection and terminal-state triage only unless USB ID, IDCODE,
  bitstream flash, and timing transcript are all present.
- **PolarFire:** SSH/precheck only unless hash-matched dispatch and timing are authenticated.
- **Extropic TSU / XTR-0:** architecture/simulation context only; no local execution target exists.

## Planned Deliverables

| Task | Deliverable |
|---|---|
| `exp5109` | `results/experiment_5109_archive_468_activate_469.json` |
| `exp5110` | `results/experiment_5110_sota_ingestion_v469.json` |
| `exp5111` | `results/experiment_5111_fover_in_domain_pool_v469.json` |
| `exp5112` | `results/experiment_5112_fover_in_domain_selector_v469.json` |
| `exp5113` | `results/experiment_5113_fover_selector_adversarial_audit_v469.json` |
| `exp5114` | `results/experiment_5114_kan_abstraction_refinement_post_wall_v469.json` |
| `exp5115` | `results/experiment_5115_graph_evidence_fover_transfer_v469.json` |
| `exp5116` | `results/experiment_5116_hubo_2dpt_sampling_reference_v469.json` |
| `exp5117` | `results/experiment_5117_taco_harm_gated_scale_v469.json` |
| `exp5118` | `results/experiment_5118_fr11_fover_residual_memory_v469.json` |
| `exp5119` | `results/experiment_5119_sota_endpoint_rootcause_v469.json` |
| `exp5120` | `results/experiment_5120_hardware_residual_telemetry_v469.json` |
| `exp5121` | `results/experiment_5121_capstone_v469.json` |

## Success Criteria

1. FoVer pool exists with n>=150, K candidates, parsed answers, `oracle@K < 1.0`, positive headroom,
   and `verifier_is_oracle=false`.
2. In-domain FoVer selector either beats tuned self-consistency with CI95 excluding 0 while label
   ablation fails, or records a clean negative that retires the immediate verifier-moat claim.
3. No KAN task repeats toy exact-MILP scale; the post-wall task changes technique and compares honestly
   against `exp5108`.
4. FR-11 promotion happens only if held-out utility and non-forgetting guards are positive; no-promotion
   remains a valid outcome.
5. Runtime and hardware tasks report clean readiness only with transcript-backed evidence; otherwise
   they write blocked/continuity artifacts with root causes.
