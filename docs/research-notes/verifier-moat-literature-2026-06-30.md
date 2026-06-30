# Verifier-moat literature ingestion - 2026-06-30

## Artifact fields
- honest_verdict: terminal prefix; success_sota_ingested_<n>_new_papers_mapped_to_phase_d.
- new_arxiv_ids: verified-real NEW arXiv IDs (http 200), NOT in the 23-paper ingested set (no fabrication -- every method cites a source).
- sota_to_phase_d_mapping: per NEW method: which PHASE D arm/direction it strengthens + the implementation delta over the current stack + the pitfall.
- next_milestone_candidates: the strongest method(s) flagged as candidate inputs for the .464 roadmap (discover->ingest->plan->experiment).
- note_path: docs/research-notes/verifier-moat-literature-<date>.md (the synthesis the planner reads).
- reliable_channel_used: sweep_clusters/sweep_semscholar + low-concurrency WebSearch/WebFetch (NOT /deep-research -- banned from the autonomous loop).
- inference_substrate: aggregation_from_upstream_artifacts (literature synthesis, no LLM inference).
- preconditions_checked: records network/sweep-helper checks; unreachable network emits blocked_.

## Reliable channel
- Used: sweep_clusters.py, sweep_semscholar.py, low-concurrency WebSearch/WebFetch.
- Not used: /deep-research.
- Semantic Scholar result: HTTP 429 was recorded and not promoted as evidence.
- Prior .462/.463 exclusions retained for continuity: arXiv:2408.15240, arXiv:2502.01989, arXiv:2502.11157, arXiv:2502.11250, arXiv:2502.14356, arXiv:2504.00891, arXiv:2504.01005, arXiv:2504.13134, arXiv:2504.16828, arXiv:2507.01951, arXiv:2508.03686, arXiv:2508.10539, arXiv:2508.16665, arXiv:2509.24460, arXiv:2510.14913, arXiv:2510.20369, arXiv:2602.24040, arXiv:2603.04304, arXiv:2605.10158, arXiv:2605.11334, arXiv:2605.18871, arXiv:2605.24005, arXiv:2605.30085, arXiv:2606.09073, arXiv:2606.19818.

## D5 conditioning
- Exp 5036 verdict: complete_moat_execution_incomplete_cascade; moat_realized=false; moat_retired_bounded=false; decision=EXECUTION-INCOMPLETE.
- .464 condition: repair D6/D4 and harden D1/D3; pivot only after clean D1+D2 nulls.

## SOTA to PHASE D mapping

### EORM small energy outcome reward verifier
- Source: arXiv:2505.14999 (https://arxiv.org/abs/2505.14999)
- PHASE D arms: D1 LoRA-EBM, D3 EBRM
- Signal: Uses an energy-based outcome reward model to rank chain-of-thought solutions with only outcome labels, reporting a 55M-parameter verifier that can select from candidate pools and generalize to unseen models.
- Implementation delta: Over Exp 5031-5036, replace the scalar D1 LoRA scorer and D3 EBRM readout with an EORM-style small energy head over frozen MuSR candidates, then rerun delta_vs_tuned_sc and second-corpus confirmation.
- Pitfall: Outcome-only labels can learn answer-shape shortcuts; the rerun needs frozen candidates, no model-id features, and the same genuine tuned-SC baseline.
- .464 candidate: flagged_for_v464 (.464): eorm_small_energy_selector_for_d1_d3

### VPR dense verifier-grounded process rewards
- Source: arXiv:2605.10325 (https://arxiv.org/abs/2605.10325)
- PHASE D arms: D2 uPRM, D6 verifier-judge cascade
- Signal: Converts symbolic or algorithmic intermediate checks into dense turn-level rewards for agentic reasoning, improving credit assignment when reliable local verification is available.
- Implementation delta: Over Exp 5031-5036, build a D2 process-reward replay that uses only oracle-distinct intermediate checks available before the final answer, then expose the same confidence as the cheap D6 router before judge calls.
- Pitfall: The method is only non-circular if the intermediate verifier is not the answer oracle; weak or domain-leaking checks would invalidate the moat claim.
- .464 candidate: flagged_for_v464 (.464): oracle_distinct_vpr_dense_process_rewards

### ProcessThinker rollout-based process rewards
- Source: arXiv:2606.11209 (https://arxiv.org/abs/2606.11209)
- PHASE D arms: D2 uPRM, D3 EBRM
- Signal: Assigns step rewards by sampling continuations from intermediate reasoning states and using empirical final-verification success, avoiding an explicit trained PRM.
- Implementation delta: Over Exp 5031-5036, compute rollout-success process scores for cached candidate prefixes and distill them into the D2 selector or D3 energy margin before comparing against tuned self-consistency.
- Pitfall: Continuation rollouts can be expensive and can leak final-answer verification into the selector; the rerun must charge compute and keep the verifier oracle-distinct.
- .464 candidate: flagged_for_v464 (.464): rollout_process_reward_distillation

### PURM reward-distribution uncertainty
- Source: arXiv:2503.22480 (https://arxiv.org/abs/2503.22480)
- PHASE D arms: D1 LoRA-EBM, D3 EBRM, D6 verifier-judge cascade
- Signal: Generalizes Bradley-Terry reward modeling to reward distributions and uses distribution overlap as per-sample uncertainty to reduce reward hacking.
- Implementation delta: Over Exp 5031-5036, turn D1/D3 scalar verifier scores into reward distributions, penalize high-overlap uncertain candidates, and route only uncertain pairs to D6 judge fallback.
- Pitfall: PURM is preference-alignment evidence, not a direct reasoning-selector result; calibration gains must not be counted unless selection accuracy improves.
- .464 candidate: flagged_for_v464 (.464): purm_uncertainty_calibrated_selector

### Consequence-Based Utility oracle-free evaluator
- Source: arXiv:2602.06291 (https://arxiv.org/abs/2602.06291)
- PHASE D arms: D6 verifier-judge cascade, D2 uPRM
- Signal: Scores a candidate solution by testing whether it improves solving of related verifiable questions, outperforming reward models, generative reward models, and LLM judges on research-level math ranking.
- Implementation delta: Over Exp 5031-5036, add a cheap consequence-evaluation branch to D6: candidate traces become exemplars for generated neighboring checks, and only low-margin cases escalate to the judge.
- Pitfall: Generating related verifiable questions can become the expensive verifier; the cascade must charge that cost and prevent neighborhood tasks from leaking answers.
- .464 candidate: flagged_for_v464 (.464): consequence_utility_cascade_pivot

## Next milestone candidates
- flagged_for_v464 (.464): eorm_purm_d1_d3_rerun: Exp 5036 D1 and D3 had positive deltas but CI touched zero. The .464 rerun should test a small EORM head with PURM uncertainty penalties before spending on a larger verifier.
- flagged_for_v464 (.464): vpr_processthinker_d2_repair: D2 was clean negative in Exp 5036. The best .464 repair is not another scalar logprob selector, but VPR-style local checks or rollout-derived process rewards evaluated on frozen candidates.
- flagged_for_v464 (.464): consequence_uncertainty_cascade: Exp 5036 was execution-incomplete because D6 and second-corpus confirmation were blocked. Consequence utility plus PURM uncertainty gives a cheap-router path that can be costed separately from judge fallback.
