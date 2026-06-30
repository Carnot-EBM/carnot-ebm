# Verifier-moat literature ingestion - 2026-06-30

## Artifact fields
- honest_verdict: terminal prefix; success_sota_ingested_<n>_new_papers_mapped_to_phase_d.
- new_arxiv_ids: verified-real NEW arXiv IDs (http 200), NOT in the 18-paper ingested set (no fabrication -- every method cites a source).
- sota_to_phase_d_mapping: per NEW method: which PHASE D arm/direction it strengthens + the implementation delta over the current stack + the pitfall.
- next_milestone_candidates: the strongest method(s) flagged as candidate inputs for the .463 roadmap (discover->ingest->plan->experiment).
- note_path: docs/research-notes/verifier-moat-literature-<date>.md (the synthesis the planner reads).
- reliable_channel_used: sweep_clusters/sweep_semscholar + low-concurrency WebSearch/WebFetch (NOT /deep-research -- banned from the autonomous loop).
- inference_substrate: aggregation_from_upstream_artifacts (literature synthesis, no LLM inference).
- preconditions_checked: records network/sweep-helper checks; unreachable network emits blocked_.

## Reliable channel
- Used: sweep_clusters.py, sweep_semscholar.py, low-concurrency WebSearch/WebFetch.
- Not used: /deep-research.
- Semantic Scholar result: HTTP 429 was recorded for later focused queries, not promoted as evidence.
- Prior .462 and planning-pass exclusions retained for continuity: arXiv:2408.15240, arXiv:2502.01989, arXiv:2502.11157, arXiv:2504.00891, arXiv:2504.01005, arXiv:2504.13134, arXiv:2504.16828, arXiv:2508.03686, arXiv:2508.10539, arXiv:2508.16665, arXiv:2509.24460, arXiv:2510.14913, arXiv:2510.20369, arXiv:2602.24040, arXiv:2603.04304, arXiv:2605.10158, arXiv:2605.18871, arXiv:2605.24005, arXiv:2606.09073, arXiv:2606.19818.

## D5 conditioning
- Exp 5022 verdict: complete_moat_execution_incomplete_ebrm; moat_realized=false; moat_retired_bounded=false; decision=EXECUTION-INCOMPLETE.
- .463 condition: repair or rerun D1/D2/D3/D6 before any retirement claim.

## SOTA to PHASE D mapping

### CoT-Entropy uncertainty-aware generative PRM
- Source: arXiv:2502.11250 (https://arxiv.org/abs/2502.11250)
- PHASE D arms: D2 uPRM, D3 EBRM
- Signal: Adds uncertainty quantification to generative reward models for step-wise verification, using CoT Entropy to detect unreliable PRM judgments.
- Implementation delta: Over Exp 5017-5022, attach CoT-Entropy uncertainty to the D2 step verifier and to the D3 EBRM selector, then require selection delta versus genuine tuned-SC after uncertainty-aware abstention.
- Pitfall: The paper is math-reasoning PRM evidence; uncertainty may flag style variation rather than wrong reasoning unless calibrated on the MuSR cache.
- .463 candidate: flagged_for_v463 (.463): cot_entropy_uprm_ebrm_uncertainty

### VERDI single-call decomposed judge confidence
- Source: arXiv:2605.11334 (https://arxiv.org/abs/2605.11334)
- PHASE D arms: D6 verifier-judge cascade
- Signal: Extracts confidence from verification sub-check traces without extra judge calls, replacing unavailable or saturated logprob confidence.
- Implementation delta: Over Exp 5017-5022, rerun the blocked D6 cascade with VERDI-style step-verdict alignment, claim margin, and evidence-grounding features as the cheap confidence router before escalating to the judge.
- Pitfall: A cascade win can silently become a judge win; the artifact must charge every fallback call and keep oracle-distinct cheap-verifier value separate.
- .463 candidate: flagged_for_v463 (.463): verdi_confidence_routed_cascade

### Reflective generative self-supervised PRM
- Source: arXiv:2507.01951 (https://arxiv.org/abs/2507.01951)
- PHASE D arms: D1 LoRA-EBM, D2 uPRM
- Signal: Uses a shared policy and process-reward interface with a lightweight scoring head and learns trajectory selection from outcome rewards without process labels.
- Implementation delta: Over Exp 5017-5022, use the reflective self-supervised PRM recipe as the D2 unblock path and, if D1 remains base-model blocked, train only a small trajectory-scoring head over cached candidates.
- Pitfall: Outcome-derived self-supervision can reproduce generator bias and reward answer-shape shortcuts; the no-model-id and oracle-distinct audits remain mandatory.
- .463 candidate: flagged_for_v463 (.463): reflective_self_supervised_prm_unblock

### CROP conformal clean-prefix certification
- Source: arXiv:2605.30085 (https://arxiv.org/abs/2605.30085)
- PHASE D arms: D3 EBRM, D6 verifier-judge cascade
- Signal: Turns any step-risk proxy into a calibrated clean-prefix certificate, then routes uncertified suffixes for repair or review.
- Implementation delta: Over Exp 5017-5022, wrap D3 energy or D6 confidence scores with a conformal prefix threshold and evaluate certified-prefix length plus answer-selection delta instead of scalar AUROC alone.
- Pitfall: Exchangeability assumptions may fail across generated candidates; over-withholding can erase the headroom that a verifier moat needs to exploit.
- .463 candidate: flagged_for_v463 (.463): crop_conformal_prefix_gate

### Full-Step-DPO self-supervised process reward
- Source: arXiv:2502.14356 (https://arxiv.org/abs/2502.14356)
- PHASE D arms: D2 uPRM
- Signal: Trains a self-supervised process reward model that scores every reasoning step instead of relying on human or GPT-4 step labels.
- Implementation delta: Over Exp 5017-5022, replace the blocked D2 logprob-cache dependency with full-step self-supervised rewards over complete candidate traces, then compare the resulting selector with tuned-SC on the same cache.
- Pitfall: It optimizes the generator as much as the verifier; using it as a selector requires a frozen-candidate evaluation to avoid claiming training lift as moat lift.
- .463 candidate: flagged_for_v463 (.463): full_step_self_supervised_uprm

## Next milestone candidates
- flagged_for_v463 (.463): repair_d2_self_supervised_prm: D5 did not cleanly retire D1 or D2. The strongest .463 repair path is a frozen-candidate self-supervised PRM that removes the blocked logprob cache.
- flagged_for_v463 (.463): rerun_d3_uncertainty_conformal_gate: D5 marks D3 execution incomplete. A .463 rerun should add CoT-entropy and CROP-style conformal thresholds before measuring delta_vs_tuned_sc.
- flagged_for_v463 (.463): verdi_oracle_distinct_cascade: D6 was blocked, but VERDI gives a single-call confidence router. .463 should test judge-call savings while preserving oracle-distinct accounting.
