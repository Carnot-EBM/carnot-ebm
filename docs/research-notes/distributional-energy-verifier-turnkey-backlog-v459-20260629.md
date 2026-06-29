# Exp 4984 Distributional Energy Verifier Turnkey Backlog Extension

- Honest verdict: `success_distributional_energy_verifier_pivot_turnkey_backlog_extended`
- Pivot executable on 7/1: `true`
- Pivot turnkey: `true`
- Three-column dry-run OK: `true`
- Moat proven claimed: `false`
- Entrypoint: `.venv/bin/python python/carnot/experiment_4984_distributional_energy_verifier_turnkey.py`

## SOTA to Carnot Mapping

### arXiv:2510.14913 - Budget-aware Test-time Scaling via Discriminative Verification
- URL: https://arxiv.org/abs/2510.14913
- Strongest method: Budget-aware discriminative verification: under a fixed inference budget, combine self-consistency with a discriminative verifier rather than spending the budget on costly generative verification traces.
- Implementation cost over current stack: Low-medium: Carnot's decomposed-energy verifier is already a discriminative learned quality scorer plus deterministic penalties. Add matched-compute accounting, report accuracy-per-cost, and compare directly against both discriminative and generative verifier frontiers.
- Pitfalls: A fixed compute budget can make an accurate verifier lose if its candidate scoring overhead is too high; hybrid self-consistency can hide whether the verifier itself adds oracle-distinct signal; math frontier results must be revalidated on TravelPlanner or MuSR.
- Roadmap input: `candidate_next_milestone: discriminative_budget_frontier`

### arXiv:2603.04304 - V1: Unifying Generation and Self-Verification for Parallel Reasoners
- URL: https://arxiv.org/abs/2603.04304
- Strongest method: V1-Infer pairwise self-verification: use uncertainty-guided tournament ranking over candidate pairs, allocating verification compute where relative correctness is most uncertain.
- Implementation cost over current stack: Medium-high: add a pairwise ranking comparator around the current candidate table, use ensemble STDDEV to target regeneration or abstention, and keep the learned quality mean as the pointwise baseline.
- Pitfalls: A unified generator/self-verifier can collapse into a model-identity shortcut, pairwise tournaments can spend substantial compute, and code or math wins may not transfer to oracle-distinct TravelPlanner/MuSR constraints without a separate verifier signal.
- Roadmap input: `candidate_next_milestone: v1_uncertainty_guided_regeneration_comparator`

### arXiv:2605.18871 - Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning
- URL: https://arxiv.org/abs/2605.18871
- Strongest method: Decomposed energy verifier: heterogeneous LoRA quality-scorer ensemble on one frozen encoder; ensemble mean ranks candidates, ensemble stddev triggers targeted regeneration or abstention; deterministic analytical constraint penalties remain separate.
- Implementation cost over current stack: Medium: keep Carnot's FoVer verifier ensemble as the analytical penalty source, add a learned LoRA-ensemble quality scorer, calibrate mean/stddev on structured rows, and prohibit model_id or oracle-label features in the verifier path.
- Pitfalls: Fails or overclaims when self-consistency is near ceiling, deterministic penalties silently become the correctness oracle, code tasks leak model identity, or stddev abstention is tuned on test labels.
- Roadmap input: `flagged_for_next_milestone: decomposed_energy_lora_ensemble_with_fover_penalties`

### arXiv:2504.16828 - Process Reward Models That Think
- URL: https://arxiv.org/abs/2504.16828
- Strongest method: ThinkPRM: a generative long-CoT process verifier that writes a step-wise verification trace and uses that generated reasoning as the reward signal for best-of-N selection or reward-guided search.
- Implementation cost over current stack: Medium-high: keep FoVer penalties as deterministic checks, add a generative PRM comparator or labeler for the learned quality-scorer ensemble, and account for verifier tokens against self-consistency at matched compute.
- Pitfalls: Verifier tokens can dominate cost, generated rationales may re-derive the generator answer instead of judging it, process labels may not transfer from math to TravelPlanner/MuSR, and long-CoT traces are hard to expose safely.
- Roadmap input: `support_for_next_milestone: thinkprm_generative_prm_comparator`

### arXiv:2502.01989 - VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model
- URL: https://arxiv.org/abs/2502.01989
- Strongest method: VFScale: train an intrinsic diffusion energy landscape with MRNCL plus KL regularization, then use hybrid MCTS over denoising trajectories so the intrinsic energy acts as a dense verifier/reward.
- Implementation cost over current stack: High: FoVer penalties can provide analytical constraints, but VFScale requires a generator-side diffusion or denoising search substrate plus dense-energy training; it is not a drop-in replacement for the current cached-candidate verifier.
- Pitfalls: Evidence is strongest on Maze/Sudoku-style diffusion reasoning, not LLM structured-output reranking; the verifier-free objective can blur the oracle-distinct control, and hMCTS cost can erase matched-compute gains.
- Roadmap input: `support_for_next_milestone: vfscale_intrinsic_energy_dense_reward_ablation`

### arXiv:2508.16665 - Trust but Verify! A Survey on Verification Design for Test-time Scaling
- URL: https://arxiv.org/abs/2508.16665
- Strongest method: Verifier-design taxonomy for test-time scaling: outcome vs process, generative vs discriminative, prompt-based vs trained, and utility axes for efficiency and abstention.
- Implementation cost over current stack: Low: no model change required. Use it to label Carnot's current decomposed-energy verifier as a discriminative outcome-ranker with analytical constraint penalties, uncertainty, and abstention controls.
- Pitfalls: It is a survey, not a direct measured win; taxonomy language can hide whether the verifier is oracle-distinct, matched-compute, or evaluated on a self-consistency-not-saturated domain.
- Roadmap input: `taxonomy_anchor: verifier_design_cell_and_adjacent_open_cells`

### arXiv:2508.10539 - Improving Value-based Process Verifier via Low-Cost Variance Reduction
- URL: https://arxiv.org/abs/2508.10539
- Strongest method: ComMCS variance reduction for value-based process verifiers: combine current-step and later-step Monte Carlo value estimates to reduce annotation variance without additional LLM inference.
- Implementation cost over current stack: Medium: add process-state value labels or cached rollouts to the TravelPlanner/MuSR candidate traces, then calibrate the learned quality-ensemble STDDEV used by the regenerate/abstain loop.
- Pitfalls: Evidence is math-centric; reducing variance can over-smooth genuine epistemic disagreement; adjacent-step value estimates may be unavailable for outcome-only rows.
- Roadmap input: `candidate_next_milestone: variance_reduced_uncertainty_head`

### arXiv:2502.11157 - Dyve: Thinking Fast and Slow for Dynamic Process Verification
- URL: https://arxiv.org/abs/2502.11157
- Strongest method: Dynamic process verification with a cheap System-1 token-level fast path and selective System-2 comprehensive analysis for hard or ambiguous steps.
- Implementation cost over current stack: Medium-high: wrap FoVer plus the learned energy scorer in a cascade router that accepts easy rows cheaply and escalates only uncertainty or constraint-conflict rows to a slower process verifier.
- Pitfalls: Router false negatives can skip needed slow checks; slow-path tokens can erase efficiency gains; ProcessBench/MATH evidence must be revalidated on TravelPlanner or MuSR before any moat claim.
- Roadmap input: `candidate_next_milestone: fast_slow_process_router`

### arXiv:2504.01005 - When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning
- URL: https://arxiv.org/abs/2504.01005
- Strongest method: Compute-optimal fixed-budget analysis of when to allocate inference tokens to additional self-consistency samples versus fewer samples plus a generative verification pass.
- Implementation cost over current stack: Low-medium: add matched-compute accounting around the existing three-column harness, then report the decomposed-energy verifier against the self-consistency/generative-verification frontier rather than only against equal candidate counts.
- Pitfalls: A verifier that improves accuracy can still lose the north-star win if verification tokens are too expensive; frontier conclusions can flip by domain, generator strength, and self-consistency saturation.
- Roadmap input: `candidate_next_milestone: efficiency_parity_frontier`

### arXiv:2504.00891 - GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning
- URL: https://arxiv.org/abs/2504.00891
- Strongest method: GenPRM: a generative process reward model that reasons over each step, emits verification chains, and can invoke code-style checks as test-time compute scales.
- Implementation cost over current stack: Medium-high: add a generative PRM comparator beside the discriminative decomposed-energy verifier, charge its reasoning and code-check tokens against the same budget, and keep FoVer penalties as the deterministic oracle-distinct analytical column.
- Pitfalls: Generated verification traces may re-derive the generator answer, code checks are not available for every TravelPlanner/MuSR-style constraint, and verifier-token cost can erase matched-compute gains.
- Roadmap input: `candidate_next_milestone: genprm_matched_compute_generative_comparator`

### arXiv:2509.24460 - ContextPRM: Leveraging Contextual Coherence for multi-domain Test-Time Scaling
- URL: https://arxiv.org/abs/2509.24460
- Strongest method: ContextPRM: multi-domain process verification that uses contextual coherence signals to scale test-time verification beyond a single math-heavy verifier domain.
- Implementation cost over current stack: Medium-high: extend the verifier registry with domain tags, coherence features, and cross-domain calibration slices, then compare against the current math-strong/code-weak Carnot verifier stack.
- Pitfalls: Context coherence can reward fluent but wrong traces, cross-domain PRM generalization may hide per-domain failures, and registry expansion must avoid turning an executable oracle into the verifier itself.
- Roadmap input: `candidate_next_milestone: contextprm_cross_domain_registry_comparator`

## Validation Gate

The post-6/30 experiment must beat self-consistency with CI95 excluding zero, remain oracle-distinct (`verifier_is_oracle=false`), avoid a model-identity shortcut, and evaluate a domain where self-consistency is not near-ceiling. Exp4984 states this gate but does not claim it has been met.
