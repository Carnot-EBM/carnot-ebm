# Exp 4940 Distributional Energy Verifier Executable Spec

- Honest verdict: `success_distributional_energy_verifier_pivot_executable_spec_ready`
- Pivot executable on 7/1: `true`
- Three-column dry-run OK: `true`
- Moat proven claimed: `false`

## SOTA to Carnot Mapping

### arXiv:2605.18871 - Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning
- URL: https://arxiv.org/abs/2605.18871
- Strongest method: Decomposed energy verifier: heterogeneous LoRA quality-scorer ensemble on one frozen encoder; ensemble mean ranks candidates, ensemble stddev triggers targeted regeneration or abstention; deterministic analytical constraint penalties stay separate.
- Implementation cost over current stack: Medium: retain Carnot's FoVer verifier ensemble as the analytical/executable penalty term, add a learned LoRA-ensemble quality scorer, calibrate mean/stddev on structured rows, and forbid model_id or oracle label features in the verifier path.
- Pitfalls: Fails or overclaims when self-consistency is near ceiling, deterministic penalties silently become the oracle, code tasks leak model identity, or stddev abstention is tuned on the test labels.
- Roadmap input: `flagged_for_next_milestone: decomposed_energy_lora_ensemble_with_fover_penalties`

### arXiv:2504.16828 - Process Reward Models That Think
- URL: https://arxiv.org/abs/2504.16828
- Strongest method: ThinkPRM: a generative long-CoT process verifier that writes a step-wise verification trace and uses that generated reasoning as the reward signal for best-of-N selection or reward-guided search.
- Implementation cost over current stack: Medium-high: keep FoVer penalties as deterministic checks, add a generative PRM comparator or labeler for the learned quality-scorer ensemble, and budget explicit verifier tokens against self-consistency at matched compute.
- Pitfalls: Verifier tokens can dominate cost, generated rationales may re-derive the generator answer instead of judging it, process labels may not transfer from math to TravelPlanner/MuSR, and long-CoT traces are hard to expose safely.
- Roadmap input: `support_for_next_milestone: thinkprm_generative_prm_comparator`

### arXiv:2502.01989 - VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model
- URL: https://arxiv.org/abs/2502.01989
- Strongest method: VFScale: train an intrinsic diffusion energy landscape with MRNCL plus KL regularization, then use hybrid MCTS over denoising trajectories so the intrinsic energy acts as a dense verifier/reward.
- Implementation cost over current stack: High: FoVer penalties can provide analytical constraints, but VFScale requires a generator-side diffusion or denoising search substrate plus dense-energy training; it is not a drop-in replacement for the current cached-candidate verifier.
- Pitfalls: Evidence is strongest on Maze/Sudoku-style diffusion reasoning, not LLM structured-output reranking; the verifier-free objective can blur the oracle-distinct control, and hMCTS cost can erase matched-compute gains.
- Roadmap input: `support_for_next_milestone: vfscale_intrinsic_energy_dense_reward_ablation`

## Executable Dry-Run

The dry-run wires `self_consistency`, `decomposed_energy_verifier`, and `oracle` on the cached TravelPlanner slice. It does not run the real benchmark and does not promote a verifier-value claim.

## Validation Gate

The post-6/30 experiment must beat self-consistency with CI95 excluding zero, remain oracle-distinct (`verifier_is_oracle=false`), and pass the no-model-identity-shortcut check. Exp4940 states this gate but does not claim it has been met.
