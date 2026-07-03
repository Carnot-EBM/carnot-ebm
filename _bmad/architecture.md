# Carnot — Architecture

**Last Reconciled:** 20260703

## Overview

Carnot is a dual-language (Rust + Python/JAX) Energy Based Model framework organized as a Cargo workspace with a companion Python package. The Rust side provides performance-critical compute; the Python side provides JAX-based research workflows and exposes Rust internals via PyO3.

*Position Paper Note:* See `docs/position_paper_nexus.md` for the latest strategic roadmap and architectural updates related to Phase 3 hardware acceleration and verification pipelines.

## System Architecture

```
carnot/
├── crates/                    # Rust workspace
│   ├── carnot-core/           # Core energy function traits, types
│   ├── carnot-boltzmann/      # Large-tier EBM
│   ├── carnot-gibbs/          # Medium-tier EBM
│   ├── carnot-ising/          # Small-tier EBM
│   ├── carnot-samplers/       # MCMC samplers (Langevin, HMC)
│   ├── carnot-training/       # Training algorithms (CD, SM, NCE)
│   └── carnot-python/         # PyO3 bindings
├── python/
│   └── carnot/                # Python package
│       ├── core/              # JAX energy functions
│       ├── models/            # Boltzmann, Gibbs, Ising in JAX
│       ├── phase3/            # Phase 3 seed — bridging Ising to Kona continuous latent space
│       │   ├── continuous_ebm.py  # ContinuousEBM, sample_continuous, sample_langevin,
│       │   │                  #   sample_energy_matching, compare_samplers, compare_minima
│       │   │                  #   (Exp 435a baseline; Exp 446 Langevin+EnergyMatching)
│       │   └── __init__.py    # Exports all 8 public symbols
│       ├── samplers/          # JAX MCMC samplers
│       │   ├── parallel_ising.py  # Parallel Ising Gibbs (checkerboard, annealing, thrml-compatible)
│       │   └── backend.py     # SamplerBackend protocol (CPU, TSU stub)
│       ├── training/          # JAX training loops
│       ├── pipeline/          # Production verify-repair pipeline (Exp 74-75)
│       │   ├── extract.py     # ConstraintExtractor: Arithmetic, Code, Logic, NL, Auto
│       │   ├── verify_repair.py  # VerifyRepairPipeline — main user API
│       │   ├── errors.py      # CarnotError hierarchy, timeouts, degradation
│       │   ├── env_autofix.py # EnvironmentAutoFix — self-inject CARNOT_FORCE_LIVE (RETRO-022)
│       │   ├── experiment_watchdog.py  # ExperimentTimeoutWatchdog — hard wall-clock cap (RETRO-003)
│       │   └── long_run_executor.py   # LongRunBenchmarkExecutor — batched checkpoint-resume (RETRO-026)
│       ├── mcp/               # Production MCP server (Exp 76)
│       │   └── server.py      # verify_llm_output, verify_and_repair, health_check
│       ├── verify/            # ComposedEnergy, ConstraintTerm, repair
│       ├── inference/         # EBM loader, composite scorer, LLM solver
│       └── bindings/          # PyO3 bridge to Rust
├── crates/carnot-constraints/ # Rust constraint verification (Exp 70)
├── examples/                  # 5 integration examples (Exp 79)
├── tests/
│   ├── rust/                  # Rust integration tests
│   ├── python/                # Python/pytest tests (21,955 items collected; 3 import errors pending)
│   └── integration/           # Full pipeline integration tests (Exp 81)
├── openspec/                  # Capability specs
├── _bmad/                     # Strategic docs
├── ops/                       # Operational docs
├── epics/                     # Epics and stories
├── research-program.md        # Declarative research goals and priorities
├── research-references.md     # Technologies and ideas for future milestones
├── research-roadmap.yaml      # Active research roadmap
└── research-complete.yaml     # Completed research milestones (113 archived through 2026.04.103)
```

## ARC-AGI-3 Harness Architecture

The current north-star architecture is a live ARC-AGI-3 hidden-game discovery agent, not a public-game replay package. `ops/north-star.md` defines the two first-class metrics as live solve-rate / official score and action/compute efficiency. Public-game replay depth remains useful development evidence, but the scored competition path is the live `E3AgentPolicy` cascade in `python/carnot/agentic/arc_competition_agent.py`.

The architecture has two distinct but coupled execution paths:

| Path | Entry point | Purpose | Claim boundary |
|---|---|---|---|
| Offline development twin | `scripts/arc_loop_solve.py` | Runs deterministic no-quota public environments, adaptered verifier-routed solves, adapter-free first contact, checkpoint refresh, and reproducibility capture. | Development proxy unless the result comes from live-agent self-discovery and passes the reproduction gate. |
| Scored live cascade | `E3AgentPolicy` via the submitted competition kernel | Runs the hidden-game `choose_action` / `is_done` loop with exploration, routing, DSL/world-model induction, trust energy for hidden-state world models, optional active probes, and model-planned execution. | Authoritative hidden leaderboard path; the registry records a current scored baseline of `0.08`, while replay scorecards are not leaderboard evidence. |

The shared search substrate is verifier-routed best-first exploration. `python/carnot/agentic/arc_graph_explore.py` provides adapter-free graph exploration with salience/candidate ordering, A*-style frontier priority, goal-energy hooks, action-effect frontier priority, MAP frontier seeding, QD sequence injection, and move-pruner hooks. `python/carnot/agentic/arc_solver_kit.py` provides the durable `OfflineSolver`: adaptered best-first search ordered by a hand or learned verifier, fresh-env branching for non-idempotent reset games, reusable primitive operators, and `reproduce()` as the executable reproduction gate. A solve only counts when the captured action labels replay through this gate.

`ops/arc_solve_registry.yaml` is the knowledge-capture mechanism. It records general gotchas, reusable primitive operators, per-game mechanics, dead ends, learned verifier checkpoints, reproduction artifacts, and live-submission provenance. As of the current registry it reports `reproducible_total_levels: 69` and `reproducible_total_games: 24`; those totals are a development proxy, not a substitute for the live hidden score. The registry also preserves the practical gotchas that make the architecture work: level progress lives on frames, fresh-env replay is required for some games, coordinates must be derived from the environment, animation/facing can be state, and same-action level transitions must be counted only through the reproduction gate.

This division is intentional: the offline twin is where mechanisms are made reproducible and reusable; the `E3AgentPolicy` cascade is where those mechanisms are tested as a hidden-game runtime discovery process.

## Key Design Decisions

### DD-01: Cargo Workspace
Each logical component is a separate crate for compile-time isolation, independent versioning, and clear dependency boundaries.

### DD-02: Trait-Based Core
`carnot-core` defines traits (`EnergyFunction`, `Sampler`, `Trainer`) that all tiers implement. This enables generic algorithms over any tier.

### DD-03: JAX for Python
JAX is chosen over PyTorch for the Python side because JAX is the first-class citizen of EBM research — its functional transform model (vmap, grad, jit) maps naturally to energy function composition.

### DD-04: PyO3 Bindings
A dedicated `carnot-python` crate exposes Rust implementations to Python via PyO3/maturin, enabling researchers to use Rust performance from familiar Python workflows.

### DD-05: Tier Separation
Each tier (Boltzmann, Gibbs, Ising) is a separate crate/module to enable independent development, testing, and deployment. Users can depend on only the tier they need.

### DD-06: Autoresearch Two-Phase Loop

Self-improvement is structured as two phases running inside the research conductor:

1. **JAX Prototype Phase.** An agent (currently a local Qwen 3 variant, orchestrated via the ZeroClaw / agntcy.org framework) proposes EBM improvements — architecture tweaks, noise schedules, activation functions — and runs them in an isolated Python/JAX sandbox. Evaluation is done against the energy landscape on a held-out validation split, so "success" is an objective energy delta, not a subjective judgment.
2. **Rust Transpilation Phase.** Proven JAX improvements are translated to Rust, benchmarked for wall-clock performance, and auto-merged only if they pass both math equivalence (Rust vs. JAX produce equal energies bit-for-bit on the same seed) and performance baselines.

**Safety guardrails baked into the loop:** immutable validation data (the held-out set cannot be overwritten by a run), hard wall-clock and memory timeouts per hypothesis (see `ExperimentTimeoutWatchdog`), and rollback energy thresholds that refuse to merge any change that worsens held-out energy.

**Why this structure works for Carnot specifically:** the dual Rust + JAX layout is uniquely suited to this loop — JAX for fast math experimentation with `vmap`/`grad`/`jit`, Rust for production performance validation and deployability. Crucially, the EBM itself provides a mathematical ground truth for evaluating proposed improvements, so the loop does not need a human judge in the critical path. The IPC path between the Python orchestrator and the Rust EBM is still open (options: gRPC, shared memory, PyO3 direct) — the decision is deferred until the transpilation phase is exercised on a real improvement candidate.

This design decision is what makes REQ-AUTO-* (autonomous self-learning) a first-class requirement rather than an afterthought. Phase 3 — the open-source EBM/EBT foundation model — depends on this loop compounding.

## PHASE D Lifecycle And Retirement

PHASE D was commissioned on 2026-06-30 as the off-ARC distributional-energy verifier-moat program: LoRA-EBM holistic scoring, uPRM-style generated-text/logprob process rewards, EBRM-style post-hoc reward refinement, and closely equivalent distributional-energy rankers against genuine tuned self-consistency on headroom-present reasoning corpora.

The lifecycle is now closed. Across seven milestones the external generated-text/logprob scorer family produced null or marginal evidence rather than a decision-grade verifier moat over tuned self-consistency. `ops/exclusion_manifest.yaml` records `phase_d_external_text_scorer_retired_exp5163_v474`, with Exp 5163 as the terminal continuation and Exp 5170 as the retirement record. `ops/verifier_gaps.md` records the 2026-07-02 retirement note and the MuSR residual: headroom existed, but the clean EBRM arm tied tuned self-consistency with delta `0.0`, CI95 including zero, and McNemar `p=1.0`.

The formal retirement date is 2026-07-02/03. Future reruns of this external-text scorer construction class require an operator override citing a genuinely different mechanism. The retirement is deliberately narrow: hidden-state/internal-representation verifiers, ARC oracle-distinct verifier work, and the FoVer production ensemble are explicitly outside the retired scope.

Architecturally, PHASE D retired one mechanism class, not the verifier thesis. The lesson carried forward is that a fluent generated-text/logprob reranker is insufficient as a moat unless it beats matched tuned self-consistency under paired confidence intervals and survives the circularity/oracle-distinctness checks.

## Hidden-State Verifier Research Frontier

Hidden-state/internal-representation verifiers are the current live verifier-research frontier precisely because they are not the retired PHASE D mechanism. They score or steer a generator's internal representations rather than reranking generated text/logprobs after the fact. `ops/verifier_gaps.md` explicitly preserves TrajSelector-style hidden-state scoring and VerifySteer/PHSV-class internal-representation probes as sanctioned open mechanisms.

The current evidence is honest and still open. Exp 5200 (`results/experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476.json`) produced a negative MMLU-Pro hidden-state probe result: the trained probe did not beat tuned self-consistency and left oracle-recoverable candidates unselected. The gap entry keeps the missing discriminator live: a candidate-internal correctness signal that separates correct traces from dense wrong-answer clusters, potentially requiring a stronger supervised probe or transformer-layer sweep once practical `output_hidden_states` access is available.

For ARC, this frontier connects to two open architectural gaps:

| Gap | Current status | Architectural implication |
|---|---|---|
| `GAP-4891` | Goal detection improved through relational target-match, but Stage-2/3/4 did not bank new levels; the binding wall is trajectory enumeration/generation. | Verifier energy can identify goal-shaped states, but the harness still needs a generator/search process that enumerates the winning trajectory. |
| `GAP-4` | Same-shape rule-application consistency has a preliminary positive via execution/program-synthesis verification, but the result is not yet statistically/decentralization complete. | The strongest selector signal is rule execution: induce a program from demos, execute it on test input, and gate candidate promotion by exact or graded consistency. |

The hidden-state program therefore remains architecturally distinct from PHASE D and is still a live path toward an oracle-distinct verifier moat, but it is not yet a production claim.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core compute (Rust) | Rust stable, ndarray, rayon |
| Core compute (Python) | Python 3.11+, JAX, Flax/NNX |
| Bindings | PyO3, maturin |
| Testing (Rust) | cargo test, cargo-tarpaulin |
| Testing (Python) | pytest, pytest-cov |
| Linting (Rust) | rustfmt, clippy |
| Linting (Python) | ruff, mypy |
| Pre-commit | pre-commit framework |
| CI | GitHub Actions |

## Data Flow

```
Training:
  Data → Sampler(energy_fn) → Gradient Estimator → Parameter Update → Model

Inference:
  Model + Noise → MCMC Sampler(energy_fn) → Samples
```

## Cross-Cutting Concerns

- **Logging**: `tracing` (Rust), `logging` (Python)
- **Serialization**: `serde` (Rust), `safetensors` (both)
- **Numerics**: `f32` default, `f64` configurable
- **Parallelism**: `rayon` (Rust), `jax.pmap` (Python)

## thrml Integration

The `parallel_ising.py` sampler provides a `parallel_sample_states` function that wraps thrml's `IsingEBM` interface. It extracts coupling matrices and biases from an `IsingEBM` instance and runs the parallel checkerboard Gibbs sampler (with optional simulated annealing) as a drop-in replacement for thrml's built-in sampling — achieving 183x speedup on CPU at 100 variables and 572x at 500 variables.

## Verification Pipeline Tiers

The LLM output verification pipeline still uses a cascade architecture where cheaper tiers run first. The current `python/carnot/pipeline/verify_repair.py` path has expanded since the 2026-05-16 table; the current operational view is:

| Tier | Name | Class / hook | Runtime role | Short-circuit / certificate behavior |
|---|---|---|---|---|
| Pre-IR | Typed + semantic grounding | `extract_typed_reasoning`, `SemanticGroundingVerifier`, `SemanticVerifierV2` | Extract typed reasoning and semantic evidence before cheap gates. | Adds semantic constraints/certificate fields; no fast-path by itself. |
| 0 | Pipeline JEPA fast path | `jepa_fast_path_predictor` / `JEPA_FAST_PATH` | Predicts low violation probability from response features before extraction. | `verified=True`, `skipped=True` when below threshold. |
| 0 | Per-call JEPA fast path | `jepa_predictor` / `FAST_PATH` | Embeds the first response tokens and skips full verification when domain risk is low. | `verified=True`, `skipped=True` when max risk is below threshold. |
| 0 advisory | HalluField / semantic energy / StreamingCoT / spectral | `HalluFieldDetector`, `semantic_energy_probe`, `tier_0g_streaming_cot`, `tier_0h_spectral` | Records thermodynamic, semantic, streaming, and spectral instability signals. | Advisory certificate only. |
| 0 violation fast path | ThinkProbe | `CarnotThinkProbe` / `THINK_PROBE_FAST_PATH` | Generative 3-step CoT verdict. | Returns a violation immediately when the probe is confident the response is incorrect. |
| 0 clean fast path | NUP | `NUPProbeV4` / `NUP_PROBE_FAST_PATH` | Contrastive energy probe for cheap likely-correct responses. | `verified=True`, `skipped=True` when risk is below threshold. |
| Router | ODAR | `FreeEnergyRouter` / `ODAR_FAST_PATH` | Fuses cheap Tier-0 evidence with expected-free-energy routing. | Fast-paths low-risk cases; missing evidence falls through. |
| Hot path | Rust arithmetic/logic | `RustVerifyPipeline` | Optional arithmetic/logic verifier when the configured domain is supported. | Falls back to Python on unsupported domains or errors. |
| Core | Constraint extraction + Ising | `AutoExtractor`, `ComposedEnergy`, metadata energy | Main verification pass over static, learned, retrieved, injected, and semantic constraints. | Produces `VerificationResult` violations and energy certificate. |
| Memory additions | Constraint memory/template/embedding | `ConstraintMemory`, `ConstraintTemplateLibrary`, `EmbeddingConstraintStore`, `IsingConstraintInjector` | Adds learned/retrieved constraints and optional spin-bias injection. | Additive only; static constraints are not removed. |
| Post-core advisory | CASAL / InterWhen / AND-compose | `CASALTier`, `InterwhenMonitor`, `and_compose_k5` | Continuous-attribute, mid-stream arithmetic, and k=5 ensemble certificates. | Advisory certificate only; does not override `verified`. |

Historical 2026-05-16 table preserved below for provenance:

| Tier | Name | Class | Cost | Signal Source | Skip Condition |
|------|------|-------|------|---------------|----------------|
| 0a | CarnotThinkProbe | `CarnotThinkProbe` | ~0 ms (CI stub) / ~50-200 ms (GPU) | Generative 3-step CoT verdict (ThinkPRM, arXiv 2504.16828) | `verdict == 'incorrect'` → skip all downstream (fast-path violation) |
| 0b | SpilledEnergyDetector | `SpilledEnergyDetector` | ~0 ms (text hash) | Per-token logit-discrepancy (arXiv 2602.18671) | `high_spill_fraction <= threshold` (confident model) |
| 0c | NUP Probe v4 | `NUPProbeV4` | ~0 ms (bigram dot product) | Contrastive energy probe; max E(incorrect)-E(correct) gap (Exp 523, AUC=1.0) | `score(response) <= nup_probe_threshold` (low energy = likely correct) |
| 0d | HallucinationBasinDetector | `HallucinationBasinDetector` | ~0 ms | Latent-space basin depth via finite-difference perturbation (Exp 521, arXiv 2604.04743) | `basin_risk_score <= basin_threshold` (deep basin = stable reasoning) |
| 0e | HalluField | `HalluFieldDetector` | ~1 ms (CPU) | Token-path ensemble partition-function variance (arXiv 2509.10753); thermodynamic instability signal orthogonal to Tiers 0b/0d | Advisory signal: `is_unstable` recorded in certificate; no short-circuit (Exp 571, AUC=0.97 synthetic) |
| 1 | SinkProbe | `SinkProbe` | ~0 ms (attention reuse) | Attention sink concentration (arXiv 2604.10697) | `mean_sink_score >= sink_threshold` |
| 2 | SC-Energy | `SCEnergyEnergyAdapter` | ~1 ms (CPU, no GPU required) | SetConsistency energy: contrastive TF-IDF + 2-layer MLP trained on coherent vs incoherent CoT pairs (AUROC >= 0.75, Exp 944/969, deployed 2026-04-28); VJEPA v2 available as explicit fallback override | `coherent_score >= sc_threshold` (high score = coherent) |
<!-- Tier 2 updated: SC-Energy (SCEnergyEnergyAdapter, AUROC>=0.75) replaces VJEPA v2 as default Tier 2 OOD detector (Exp 1001, milestone .78, 2026-04-28); VJEPA v2 (OOD AUC=0.9211, Exp 884) remains available as explicit override via _load_jepa_model() -->
<!-- Tier 2 history: VJEPA v2 ood_auc=0.9211 (Exp 884, milestone .68, 2026-04-26); prior Tier 2 was EORMModel (55M-param CoT energy reward model, trained in Exps 340/341/355/359) -->
| 2.5 | SymCodeVerifier | `SymCodeVerifier` | ~1-5 ms/step (regex) / ~100-500 ms/step (LLM) | Executable Python verification of CoT arithmetic; distribution-invariant (Exp 619, AUC=0.804 live) | `detection_score == 0.0` (no arithmetic violations) |
| 2.6 | HermesVerifierAdapter (candidate) | `HermesVerifierAdapter` | ~1-5 ms/step (CI regex) / ~100-500 ms/step (LLM) | HERMES step-boundary feedback loop (arXiv 2511.18760): LLMAsExtractorV1 translator + SymCodeVerifier prover + correction hint injection before next step (Exp 633, CPU prototype) | `prover_verdict == 'correct'` for all steps |
| 2.7 | CausalReasoningVerifier | `CausalReasoningVerifier` | ~1 ms/step-pair (regex) | Causal entailment checking across CoT step boundaries (arXiv 2601.21210): orthogonal to Tier 2.5 arithmetic checking. Detects "correct arithmetic, wrong carry-forward" errors. (Exp 642, causal_recall=0.36 > baseline=0.12) | `any_violation(response) == False` |
| 3 | Ising | `VerifyRepairPipeline` | ~0.006 ms/constraint | Full constraint verification | Always runs if tiers 0-2 pass |

 Tier 2 updated to SC-Energy (SCEnergyEnergyAdapter, AUROC>=0.75) by Exp 1001 on 2026-04-28; prior Tier 2 was VJEPA v2 (VariationalJEPAPredictor, OOD AUC=0.9211, Exp 884, 2026-04-25); prior to that was EORMModel (55M-param CoT energy reward model, Exps 340/341/355/359). Each tier returns early if it can clear the response, avoiding subsequent more expensive tiers. Tier 0a (CarnotThinkProbe) was added in Exp 444 (arXiv 2504.16828, ThinkPRM). Tier 0b was added in Exp 433 (arXiv 2602.18671, ICLR 2026). Tiers 0c and 0d were wired in Exp 530 (REQ-VERIFY-111, REQ-VERIFY-112). Tier 0e (HalluField) was added in Exp 571 (arXiv 2509.10753) as an advisory thermodynamic instability signal. Tiers 1-3 were designed in Exps 346-348/360. Tier 2.5 (SymCodeVerifier) was added in Exp 619 (REQ-VERIFY-122, AUC=0.804 live). Tier 2.6 (HermesVerifierAdapter, candidate) was prototyped in Exp 633 (REQ-VERIFY-136, arXiv 2511.18760 HERMES) as a step-boundary feedback adapter using LLMAsExtractorV1 + SymCodeVerifier. Tier 2.7 (CausalReasoningVerifier) was added in Exp 642 (REQ-VERIFY-139, arXiv 2601.21210) as an orthogonal causal-break detector checking numeric carry-forward across step boundaries.

CarnotThinkProbe is optional in VerifyRepairPipeline.verify() — pass `think_probe=CarnotThinkProbe(llm_caller=caller)` to enable. When enabled: if verdict='incorrect', returns VerificationResult(verified=False, mode='THINK_PROBE_FAST_PATH', skipped=True) without running Ising. CI stub (llm_caller=None, the default) returns 'uncertain' and falls through to Ising.

## KAN Fast-Path Tier — KAEMEnergy (Exp 447)

`python/carnot/models/kaem_energy.py` adds a **KAEMEnergy** model that replaces the iterative Ising/Gibbs MCMC inner loop with exact inverse-transform sampling (arXiv 2506.14167, June 2025).

| Component | Class | Key property |
|-----------|-------|--------------|
| Per-variable splines | `UnivariateKAEMLayer` | Energy = sum_i e_i(x_i); variables are independent under Gibbs distribution |
| Marginal CDF | `marginal_cdf(var_idx, x)` | Numerical integration (trapezoidal, 256-point grid) |
| Exact sampling | `sample_exact(n_samples, rng_key)` | Inverse-transform via binary search on precomputed CDF table; O(log 256) per variable |
| Full model | `KAEMEnergy` | energy() / sample() / fit(); no MCMC required |
| Benchmark | `benchmark_kaem_vs_mcmc` | Wall-clock comparison vs ParallelIsingSampler; returns speedup_ratio |

**Why this matters:** The Kolmogorov-Arnold theorem decomposes the energy into univariate per-variable terms. Each marginal CDF can be computed and inverted in closed form — no burn-in, no autocorrelation. Target speedup: 10-100x for sub-100-variable problems. Hardware path: bisection is pure arithmetic, FPGA-native. Exported from `carnot.models.__init__`.

Spec: REQ-SAMPLE-015, REQ-SAMPLE-016, SCENARIO-SAMPLE-027/028/029.

## Asymptotic Hardware Mandate (Phase 2 → Phase 3)

**Core insight:** Hardware acceleration via the continuous-to-Ising transpiler is **not a performance optimisation** for the EBM verifier. It is a **mathematical requirement** for self-distillation to avoid divergent sample budgets as the model converges.

### The stochastic bottleneck

In a deterministic framework, error contracts geometrically. With finite-sample noise from MCMC (PT-PCD / Gibbs), the variance scales as $\sigma^2 \tau_{\text{int}} / N$ where $\tau_{\text{int}}$ is integrated autocorrelation time. To maintain strict contractivity (signal exceeds noise floor):

$$
N > \frac{\sigma^2 \, T \, \tau_{\text{int}}}{2 \, \Phi \, \delta_t}
$$

where $T$ = filter temperature, $\Phi$ = structural restorative force in $L^2(\mu_P)$, $\delta_t$ = current model error. Derivation in `docs/research-notes/zenil-alpha-verifier-derivation.md`.

### Implications

1. **Asymptotic divergence.** As $\delta_t \to 0$, required $N$ diverges to infinity.
2. **Phase-transition blowup.** At critical points in the energy landscape, $\tau_{\text{int}} \to \infty$, further exploding $N$.
3. **The Phase 2 mandate.** CPU/GPU caps at $\sim 10^6$ samples/sec — sufficient for $\delta_t \gtrsim 10^{-3}$. Phase 3 foundation-model regime requires $\delta_t \to 10^{-9}$ or beyond, which mathematically forbids the $10^6$ ceiling.

### Hardware path

| Phase | Hardware | Samples/sec | Convergence depth |
|-------|----------|-------------|-------------------|
| 1 | CPU / ROCm | $10^6$ | $\delta_t \gtrsim 10^{-3}$ |
| 2 | KV260 / FPGA Ising | $10^9$ | $\delta_t \gtrsim 10^{-6}$ |
| 2+ | Extropic XTR-0 (TSU) | $10^{11}$–$10^{12}$ | $\delta_t \gtrsim 10^{-9}$ |
| 3 | photonic / Ising-machine cluster | $\gtrsim 10^{13}$ | foundation-model regime |

### Sampler-Optimization Decision Record

SpecAnn rejected for Phase 3 inference-time argmin. Rationale: (a)
HUBO→QUBO reduction injects gadgets+penalties that fracture SpecAnn's spectral
homotopy path; (b) phase-transition level-crossings during training force
catastrophic cold-restarts; (c) three-paper composition
(SpecAnn+BRAIN+MCMC Layers) triggers Gadget-Induced Mean-Field Collapse (Deep
Think DT-COMPOSITION (f), 2026-05-08). Carnot retains existing
Gibbs-heuristic argmin on unreduced HUBO energy.

### Active hardware tracks (Exp 1460)

As of the 20260507 scope-reduction decision, Carnot keeps exactly three
active hardware tracks. "Active" means a milestone may spend work on the
track now without adding another speculative branch; it does not imply
that hardware execution has happened.

### 2026-07-03 Hardware Continuity Update

The current hardware state is narrower than the Exp 1460 portfolio table:

| Board | Current status | Architectural implication |
|---|---|---|
| KV260 | Near-terminal / terminal-continuity board. Exp 5201 checked it by SSH only (`ssh -o ConnectTimeout=5 -o BatchMode=yes kria true`) and recorded a hash-verified smoke with no speedup claim. | Keep as the sovereignty-story anchor for edge deployability and hardware-energy-evaluator continuity, but do not infer speedup or inspect host block devices as a KV260 precondition. |
| PolarFire | Reachable by SSH and hash-verified smoke in Exp 5201, but `polarfire_workload_validated=false`; the terminal bar remains an end-to-end Carnot dispatch run. | Opportunistic continuity only; reachability is not a terminal workload claim. |
| GateMate | Blocked on DirtyJTAG / GM1Ax IDCODE detection. Exp 5201 narrowed the failure to `jtag_protocol_level`: USB enumeration, permissions, tool version, and 100 kHz..15 MHz clock-rate hypotheses were mechanically eliminated; the leading untested physical hypothesis is cable/port/JTAG-side wiring or board power. | Keep the blocker visible, but do not burn milestones on repeated identical `openFPGALoader -c dirtyJtag --detect` runs without a new operator-side physical angle. |

The historical Exp 1460 active/deferred tables remain below as portfolio provenance; the dated row above is the current operational state.

| Active track | Current scope | Claim boundary |
|---|---|---|
| Dual RTX 3090 CUDA local SOTA runtime repair | Repair the local GGUF/llama.cpp CUDA runtime for cached Qwen/Gemma models visible in Exp 1442. | No live SOTA inference claim until a smoke run records `usable_response=true` on the target runtime. |
| KV260/FPGA Discrete SB RTL lint and simulation | Continue source-level RTL, Verilator/Icarus/Yosys lint, and simulator evidence from Exp 1451. | No KV260 board, bitfile, or latency claim until Vivado synthesis, bitfile flashing, and board commands are captured. |
| THRML/Extropic TSU compatibility simulation | Keep TSU readiness at the THRML/JAX compatibility, sampler-interface parity, and CPU-simulation layer. | No Extropic hardware access, Z1/XTR-0 execution, or TSU latency claim until an authenticated hardware run is captured. |

### Deferred hardware tracks (Exp 1460)

These tracks remain architecturally relevant but are out of active scope
until their reopen condition is met.

| Deferred track | Reason deferred | Reopen condition |
|---|---|---|
| KV260 board execution and latency claims | Vivado is absent, no bitfile was produced, and no board commands ran. | Vivado synthesis produces a bitfile, `CARNOT_KV260_BITFILE` points to it, and a KV260/PYNQ board run records real latency. |
| AMD Strix/XDNA NPU acceleration | VitisAI and IRON paths remain blocked by missing packages or wheels. | `mlir-aie` or AMD's VitisAI onnxruntime wheel is installed and a local NPU benchmark reports real speedup. |
| Extropic Z1/XTR-0 hardware execution | No local Extropic hardware or authenticated execution transcript exists. | Carnot has early-access credentials or hardware and a THRML/SDK run records model, device, latency, and sample-quality evidence. |
| Photonic or optical Ising-machine substrates | No local optical hardware or provider run exists. | A concrete photonic provider, simulator-to-hardware API, or collaborator run can evaluate Carnot Ising cases. |
| D-Wave QPU cloud experiments | Cloud QPU access is not the current blocker for repair/runtime evidence. | A specific Ising/QUBO benchmark cannot be answered by CPU, GPU, or KV260 simulation and a Leap token plus budget are available. |
| Large production FPGA boards | Production FPGA purchases do not help until the KV260 RTL path closes. | KV260 lint, synthesis, and board execution produce a measured sampler result that justifies larger fabric. |
| RX 7900 XTX Thunderbolt eGPU path | The local CUDA RTX 3090 pair is currently more ready for SOTA runtime repair. | The RTX CUDA path is exhausted or ROCm/JAX on the eGPU is connected and verified with a real Carnot benchmark. |

### Mitigations against $\tau_{\text{int}}$ blowup

The continuous-to-Ising transpiler ships three structural mitigations against autocorrelation explosion at phase transitions:

- **Gray-code visible-spin encoder** (`python/carnot/hardware/transpiler/gray_code.py`): Hamming-distance-1 between adjacent quantization cells eliminates artificial barriers from binary-encoding cliffs.
- **Persistent Parallel Tempering with replica exchange** (`python/carnot/hardware/transpiler/distill.py`): high-T mode discovery bridges to low-T precision via swap acceptance.
- **In-loop $\tau_{\text{int}}$ diagnostics** (`python/carnot/hardware/transpiler/diagnostics.py`): explicit autocorrelation tracking flags mixing degradation before contractivity is lost.

### Decentralisation-respecting consequence

This argument compounds with rule 5 (hardware portability as political requirement). Sovereign access to high-throughput EBM sampling is a *prerequisite* for sovereign Phase 3 training, not an optional accelerator. Carnot's KV260/Extropic/photonic tracks are sovereignty infrastructure on this argument.

Cross-refs:
- `docs/research-notes/zenil-alpha-verifier-derivation.md` (full derivation)
- `memory/project_zenil_alpha_grounding.md`
- `research-references.md` → "Zenil 2026 self-improvement limits"
- `openspec/change-proposals/continuous-to-ising-transpiler.md` (Phase 2 module)

## Phase-3 → Phase-7 Defence-Layer Stack

**Last reconciled:** 2026-04-29 (after a six-round Deep Think
derivation chain). Canonical reference for the active architectural
defence layers. Publication framing in
`docs/position-paper-outline.md`; per-round derivations in
`docs/research-notes/*-deep-think-*.md`.

### Stack overview

| Phase | Threat | Mechanism | Status |
|---|---|---|---|
| 3 | Static specification gaming | Rotation defence + AND-composition + transversality | Closed-form bounds |
| 4 | Concept drift | Factorized per-verifier curriculum + UCM + DVS | Closed-form bounds |
| 5 | Detection latency | Predictive Local Linear Trend UCM | Information-Action Bottleneck |
| 6 | Whip Attack + Shadow Boundary | Multi-scale half-octave ensemble + $\theta_F^*$-rejection + Manifold Substitution | Closed-form bounds |
| 7 (proposed) | Cyclic Recurrence (FIFO Churn Gap) | Continuum memory (Hope-inspired) | Pending Deep Think Round-7 |

### Key closed-form theorems (canonical)

**Round-12 (corrected) saturation** (under $Z_t < 1$ normalization):
$\delta_\infty = C_Z \cdot \|\nu_0^\perp\|$, $C_Z = \prod_t Z_t^{-1} > 1$.

**Curriculum exponential gain:** $\Delta C_Z \propto \exp(\|\nu_0^\parallel\|^2)$.

**DVS quality threshold:** $\Lambda^* = Z_{k+1}$.

**PAC-Bayes audit budget:** $K^* = \tilde{O}((d+\log(1/\delta))/Z_{k+1}^2)$.

**Information-Action Bottleneck (Phase-5):**
$\Delta_{\text{lat}}^{\min} = \dot{\rho}(\tau_{\text{action}} - \tau^*)^+ + z\sigma_{\text{pred}}(\tau^*)$.

**Optimal LLT window:** $W^* = (72 \sigma_{\hat{\rho}}^2/\ddot{\rho}^2)^{1/5}$.

**Phase-6 multi-scale parameters:**
- Spacing $b^* = \sqrt{2}$ (half-octave; dyadic leaks 68% Whip evasion)
- Per-scale confidence $z_{1-\delta_m}^* \approx \sqrt{C - 2\ln(W_m)}$ (looser at longer scales)
- Base $W_0^* = (12 z_0^2 \sigma_{\hat{\rho}}^2 / (f_s \dot{\rho}_{\max}^2))^{1/3}$

**Geometric transversality floor:**
$\theta_F^* \approx k \sigma_{\max} \sqrt{\tau_1 / \tau_{\text{mix-budget}}}$
— linear in $k\sigma_{\max}$, not inverse.

**Phase-6 saturation:**
$\delta_\infty^{\text{Phase-6}} = C_Z [\Delta_{\text{churn}} + (12 z_0^2 \sigma_{\hat{\rho}}^2 \dot{\rho}_{\max}/f_s)^{1/3} + z_{M-1}^* \sigma_{\text{pred}}(W_{M-1})]$.

### Hardware portability theorem

> *"Provided individual verifier constraint manifolds intersect
> transversally ($\theta_F > 0$), Carnot's parallel-tempered
> AND-composition architecture guarantees strictly polynomial MCMC
> sampling latency across discrete FPGA Glauber dynamics, continuous
> thermodynamic samplers, and optical photonic substrates."*

Substrate-specific deployment:

| Substrate | Max $k$ | Topology | Bit-width |
|---|---|---|---|
| KV260 / VU9P FPGA | 4–5 | Parallel PT-SB chains | 8–16 bit/chain |
| Extropic XTR-0 | 15+ | Continuous thermodynamic | Analog (no $\kappa$) |
| Photonic Ising | 15+ | Optical interference | Speed-of-light additive |

**Round-9's $k=15$ was hardware-infeasible on single-chain FPGA.**
Geodesic-convexity is *not* preserved under AND-composition;
heterogeneous strictness creates $\kappa = \sigma_{\max}/\sigma_{\min}$
ill-conditioning. Pivot: $k\leq 5$ + Manifold-Substitution + PT-SB
parallel chains on FPGA; $k=15$ acceptable on Extropic / photonic.

### Operational implications

1. **KV260 experiments target $k_{\max} = 5$** (re-scope before
   FPGA prototype tape-out).
2. **DVS enforces three acceptance criteria** ($\Lambda^*$, $K^*$,
   $\theta_F^*$).
3. **Manifold Substitution** replaces FIFO eviction when $\theta_F^*$
   check fails (zero audit inflation).
4. **UCM is multi-scale** — single-window predictive UCM is *worse
   than reactive* against slow-stealth.
5. **Curriculum schedule smoothed** near bang-bang $\sqrt{C}$ to
   avoid first-order phase transition's exponential re-mixing.

### Cross-refs

- `docs/position-paper-outline.md` — publication framing
- `docs/research-notes/concept-drift-deep-think-results.md` (Round-13)
- `docs/research-notes/round12-renormalization-deep-think-results.md`
- `docs/research-notes/dvs-quality-curriculum-deep-think-results.md`
- `docs/research-notes/predictive-ucm-deep-think-results.md`
- `docs/research-notes/and-composition-mixing-deep-think-results.md`
- `docs/research-notes/phase6-ensemble-thetaF-deep-think-results.md`
- `docs/research-notes/phase7-continuum-memory-deep-think-prompt.md` (capstone, pending)
- `docs/research-notes/nested-learning-hope-relevance.md`
