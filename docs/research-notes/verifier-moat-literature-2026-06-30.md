# Verifier-moat literature ingestion - 2026-06-30

## Artifact fields
- honest_verdict: terminal prefix; success_sota_ingested_5_new_papers_mapped_to_phase_d.
- new_arxiv_ids: verified-real NEW arXiv IDs (http 200), NOT in the 13-paper ingested set (no fabrication -- every method cites a source).
- sota_to_phase_d_mapping: per NEW method: which PHASE D arm it strengthens (D1/D2/D3) + the implementation delta over the current stack + the pitfall.
- next_milestone_candidates: the strongest method(s) flagged as candidate inputs for the .462 roadmap (discover->ingest->plan->experiment).
- note_path: docs/research-notes/verifier-moat-literature-<date>.md (the synthesis the planner reads).
- reliable_channel_used: sweep_clusters/sweep_semscholar + low-concurrency WebSearch/WebFetch (NOT /deep-research -- banned from the autonomous loop).
- inference_substrate: aggregation_from_upstream_artifacts (literature synthesis, no LLM inference).
- preconditions_checked: records network/sweep-helper checks; unreachable network emits blocked_.

## Reliable channel
- Used: sweep_clusters.py, sweep_semscholar.py, low-concurrency WebSearch/WebFetch.
- Not used: /deep-research.
- Semantic Scholar result: HTTP 429 was recorded, not promoted as evidence.

## Phase D status read
- Exp 5007 records D1 as a flagged skeleton, D2 as blocked on logprob cache, D3 as a clean MuSR tie with tuned-SC, and the moat as scoped rather than retired.

## SOTA to PHASE D mapping

### UARM calibrated uncertainty reward head
- Source: arXiv:2606.19818 (https://arxiv.org/abs/2606.19818)
- PHASE D arms: D3 EBRM, D1 LoRA-EBM
- Signal: Adds conformal uncertainty and heteroscedastic reward variance to avoid over-weighting unreliable RM scores.
- Implementation delta: Rerun the clean D3 MuSR selector with an uncertainty head over the EBRM score, then penalize high-variance candidates before comparing against tuned self-consistency. If D1 is retried, attach the same calibration head to the LoRA-EBM scorer.
- Pitfall: The evidence is RLHF/preference-domain calibration, not a direct reasoning-selector win; style or safety uncertainty could be mistaken for reasoning correctness.
- .462 candidate: flagged_for_v462 (.462): uarm_uncertainty_head_for_d3_ebrm

### Distributional pessimistic reward uncertainty
- Source: arXiv:2606.09073 (https://arxiv.org/abs/2606.09073)
- PHASE D arms: D3 EBRM
- Signal: Frames the right reward object as a distribution p(r|x,y), with pessimistic log-moment aggregation for uncertain regions.
- Implementation delta: Replace the scalar D3 energy readout with a reward-distribution head and sweep pessimistic beta values on the existing MuSR candidate cache before any second-corpus spend.
- Pitfall: This is a unifying objective, not a finished verifier; excessive pessimism can abstain away the exact headroom D3 needs to capture.
- .462 candidate: flagged_for_v462 (.462): distributional_pessimistic_ebrm_head

### RewardUQ calibration harness for verifier uncertainty
- Source: arXiv:2602.24040 (https://arxiv.org/abs/2602.24040)
- PHASE D arms: D1 LoRA-EBM, D3 EBRM
- Signal: Compares uncertainty quantification methods for reward models and ranks them by accuracy plus calibration.
- Implementation delta: Insert a RewardUQ-style calibration table into the D1/D3 harness: ECE, AUROC on correct-vs-incorrect candidates, and selection delta after uncertainty-aware abstention.
- Pitfall: Better calibration can still leave selection accuracy tied with SC; the .462 gate must require delta_vs_tuned_sc, not calibration alone.
- .462 candidate: flagged_for_v462 (.462): rewarduq_calibration_gate_for_d1_d3

### Uncertainty-routed RM plus strong-judge cascade
- Source: arXiv:2510.20369 (https://arxiv.org/abs/2510.20369)
- PHASE D arms: D1 LoRA-EBM, D2 uPRM, D3 EBRM
- Signal: Routes uncertain preference pairs from a cheap RM to a stronger judge, improving cost-quality tradeoffs over random judge calls.
- Implementation delta: Add a matched-compute cascade control beside D1/D2/D3: cheap verifier selects when confident, uncertain pairs go to the same LLM-judge budget already used as a comparator.
- Pitfall: The judge can become the real verifier if cost and oracle-distinct boundaries are not charged explicitly; a win must separate cheap verifier value from judge fallback value.
- .462 candidate: flagged_for_v462 (.462): uncertainty_routed_moat_cascade

### LC-ERD endogenous reward decomposition
- Source: arXiv:2605.24005 (https://arxiv.org/abs/2605.24005)
- PHASE D arms: D2 uPRM
- Signal: Mines latent logic and decomposes step utility from consistency signals when explicit process labels are scarce.
- Implementation delta: Use LC-ERD as the D2 unblock path when next-token logprob caches are missing: derive process utility from consistency-regulated latent logic across the existing candidate batch, then compare to tuned SC.
- Pitfall: Endogenous consensus can preserve generator bias and create a model-identity shortcut, so the no-model-id adversarial check stays mandatory.
- .462 candidate: flagged_for_v462 (.462): lc_erd_uprm_unblock_path

## Next milestone candidates
- flagged_for_v462 (.462): distributional_uncertainty_d3_rerun: D3 is the only clean Phase D row so far and tied tuned-SC; the new papers directly address uncertainty and reward-distribution scoring.
- flagged_for_v462 (.462): lc_erd_or_logprob_cache_d2_unblock: D2 blocked on logprob candidate cache; LC-ERD supplies a process utility fallback that still must pass the model-identity shortcut audit.
- flagged_for_v462 (.462): uncertainty_routed_judge_control: If no cheap arm beats tuned-SC alone, a routed cascade can measure whether verifier uncertainty saves judge calls without relabeling the judge as the moat.
