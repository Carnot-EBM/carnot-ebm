# SOTA ingestion 2026-06-15: set-encoder and offline RFT map for .394

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_set_encoder_offline_rft_mapped_v394`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Set-LLM permutation-invariant set architecture`, arxiv_id_or_url: `2505.15433`, url: `https://arxiv.org/abs/2505.15433`}
  - {name: `AggLM review-reconcile-synthesize aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `ARBITER conservative evidence over vote prior`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `Budget-aware discriminative verification hybrid`, arxiv_id_or_url: `2510.14913`, url: `https://arxiv.org/abs/2510.14913`}
  - {name: `RAFT rejection-sampled reward-positive SFT`, arxiv_id_or_url: `2504.11343`, url: `https://arxiv.org/abs/2504.11343`}
  - {name: `VAR offline reward-weighted alignment`, arxiv_id_or_url: `2502.11026`, url: `https://arxiv.org/abs/2502.11026`}
  - {name: `Spurious Rewards same-base random-label control`, arxiv_id_or_url: `2506.10947`, url: `https://arxiv.org/abs/2506.10947`}
  - {name: `SCOPE fine-grained per-region evidence`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v394: `agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner, conditioned on the A3/A4/B2 outcomes.

## Fresh-pass provenance

Read `research-references.md` `.393 planning sweep` and `.392 planning sweep`,
`research-studying.md`, `results/experiment_4245_arc_set_encoder_beats_vote.json`,
`results/experiment_4246_code_oracle_distinct_replication.json`,
`results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json`,
and `results/experiment_4248_verifier_as_reward_offline_3arm.json`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "Set-LLM permutation invariant LLM majority vote ARC verifier aggregation" --limit 8`
- `python3 scripts/sweep_semscholar.py "reward weighted SFT verifier reward offline RAFT VAR spurious rewards" --limit 8`

The cluster helper emitted the same two broadened arXiv API URLs used by the
reliable discovery channel. Semantic Scholar returned HTTP 429 for both
focused queries, so no Semantic-Scholar-only promotion is claimed. Low
concurrency WebSearch/WebFetch verified arXiv:2505.15433, arXiv:2509.06870,
arXiv:2605.26172, arXiv:2510.14913, arXiv:2504.11343, arXiv:2502.11026,
arXiv:2506.10947, and arXiv:2512.15146.

## Exp 4245 ARC A3, Exp 4246 code A4, and Exp 4248 offline B2 read

Exp 4245 produced the first clean ARC oracle-distinct win:
`headline_outcome=arc_oracle_distinct_set_encoder_beats_vote`,
`set_encoder_minus_vote_delta=0.4423076923`, CI95 `[0.3076923077, 0.5961538462]`,
`set_encoder_minus_vote_delta` excludes zero, `margin_override_minus_vote=0.4230769231`,
`matched_control_delta=0.4807692308`, `oracle_at_k=0.8269230769`,
`held_out_task_n=52`, and `oracle_distinct_beats_vote=true`. The read is
decision-grade for ARC selection: the grown-pool set encoder beat vote and beat
the matched no-verifier control.

Exp 4246 did not replicate or refute the code oracle-distinct win. It ended as
`blocked_code_second_corpus_missing` because no cached second code candidate
pool was both hidden-label viable and source-distinct from Exp 4233. Therefore
code remains a robustness read, not a negative result against the ARC A3 win.

Exp 4248 did not run the offline reward-weighted A-vs-B comparison. It ended as
`blocked_gate_check_failed` because Exp 4247 reported `harness_smoke_passed=false`,
`steps_run=0`, `trainable_param_count=0`, and no loss movement. The B2 pivot is
still owed: fix the harness first, then run same-base Arm A verifier labels
against Arm B random labels.

## SOTA -> experiment mapping

## Set-LLM: scale the proven set architecture

**Method/source:** Set-LLM, arXiv:2505.15433
(https://arxiv.org/abs/2505.15433), adapts pretrained LLMs for permutation
invariant mixed set-text inputs.

**Carnot stack mapping:** Use it as the high-capacity version of the Exp 4245
DeepSets-style selector after the CPU-fast set encoder proved the mechanism.

**A3 ARC mapping:** Exp 4245 already landed the set-aware selector win. Set-LLM
is a scale-up baseline over a bigger pool, not the strongest new .394 idea.

**A4 code mapping:** Exp 4246 was blocked by missing corpus evidence, so Set-LLM
does not make a cross-domain code robustness claim.

**B2 reward mapping:** Orthogonal to offline reward-weighted SFT.

**Failure mode:** Multiple-choice set-text evidence does not by itself solve
free-form ARC grid synthesis.

**Experiment mapping:** Compare DeepSets selector, Set-LLM-style selector, vote,
and matched control on a bigger ARC pool.

## AggLM: synthesize a corrected grid

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), trains an aggregator to review, reconcile,
and synthesize a final answer from candidate solutions.

**Carnot stack mapping:** Add a generative reconciler after the Set-Encoder. It
should read the ranked candidates and SCOPE per-region evidence, then synthesize
a corrected grid rather than only choose an existing candidate.

**A3 ARC mapping:** The A3 win makes this the strongest .394 method: selection
works, so the next step is synthesis for cases where correct evidence is split
across candidate families.

**A4 code mapping:** Keep the AggLM arm ARC-first until a source-distinct code
pool exists.

**B2 reward mapping:** Do not mix it with the blocked reward-training claim; any
future generated training data still needs same-base A-vs-B controls.

**Failure mode:** Synthesis can fabricate grids. Exact ARC validation, vote,
selector-only, and matched-budget controls are mandatory.

**Experiment mapping:** Run an AggLM-style ARC reconciler that synthesizes a
corrected grid from Set-Encoder evidence, with a bigger pool and SCOPE
per-region ablation.

## ARBITER: diagnose wrong-majority recovery

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), frames majority-vote failures as reasoning
basins where the most stable basin can be wrong.

**Carnot stack mapping:** Record vote basin, Set-Encoder margin, and bounded
evidence-over-prior for each wrong-majority recovery.

**A3 ARC mapping:** Exp 4245's margin override also won, so ARBITER is the
right diagnostic language for which wrong-majority basins were recovered.

**A4 code mapping:** No code-basin replication can be claimed while A4 is
blocked.

**B2 reward mapping:** Not a reward-training result.

**Failure mode:** Hidden-state variants would expand the claim surface. Keep the
Carnot arm output-evidence-only unless explicitly gated.

**Experiment mapping:** Add basin-level recovery accounting to the AggLM and
Set-Encoder comparison.

## budget-aware discriminative verification: keep the vote prior

**Method/source:** Budget-aware discriminative verification, arXiv:2510.14913
(https://arxiv.org/abs/2510.14913), supports a practical hybrid of
self-consistency and discriminative verification.

**Carnot stack mapping:** Preserve vote as prior and add learned verification
only when the margin can change an answer.

**A3 ARC mapping:** Exp 4245 reports both selector lift and matched-control
lift, so .394 should add cost-normalized hybrid accounting rather than replacing
vote wholesale.

**A4 code mapping:** Robustness across code remains blocked.

**B2 reward mapping:** Selection-time efficiency is separate from reward SFT.

**Failure mode:** Hybrid scores can hide weak verifiers behind vote. Keep
selector-only and vote-only rows.

**Experiment mapping:** Match candidate and token budgets across vote,
Set-Encoder, margin hybrid, and AggLM synthesis.

## RAFT and VAR: keep offline reward weighting owed, not headline

**Method/source:** RAFT, arXiv:2504.11343
(https://arxiv.org/abs/2504.11343), and VAR, arXiv:2502.11026
(https://arxiv.org/abs/2502.11026), support offline reward-positive or
reward-weighted SFT instead of live online RL.

**Carnot stack mapping:** They remain the right shape for the offline B path:
bounded SFT over precomputed A/B/C corpora after the harness proves real
training.

**A3 ARC mapping:** They do not explain the ARC A3 selection win and should not
be flagged over AggLM for .394.

**A4 code mapping:** They reuse the stable code corpora for B2, but A4 still
needs a distinct second code candidate source for robustness.

**B2 reward mapping:** B2 blocked before training, so the next refinement is
harness repair: supported LoRA modules, at least 20 optimizer steps,
loss_final<loss_initial, and trainable_param_count>0.

**Failure mode:** Offline SFT can look stable while learning generator artifacts.

**Experiment mapping:** Once the harness passes, run VAR/RAFT-style Arm A
verifier-certified vs Arm B same-generator random labels and Arm C gold.

## Spurious Rewards: preserve the same-base random-label ablation

**Method/source:** Spurious Rewards, arXiv:2506.10947
(https://arxiv.org/abs/2506.10947), shows random rewards can recover much of an
RLVR gain on some models and are model-family dependent.

**Carnot stack mapping:** The A-vs-B reward task must use the same non-Qwen
base, same generator, and same step budget.

**A3 ARC mapping:** Not required for the test-time ARC A3 claim.

**A4 code mapping:** Does not solve the second-corpus gap.

**B2 reward mapping:** This is the non-negotiable B2 control once training
actually runs.

**Failure mode:** Without same-base random labels, a positive reward-weighted
result is uninterpretable.

**Experiment mapping:** Keep Qwen training forbidden and require Arm A minus Arm
B CI95 excluding zero.

## SCOPE: add per-region evidence

**Method/source:** SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), replaces flat majority pseudo-labels with
fine-grained, subgroup-specific confidence signals.

**Carnot stack mapping:** Convert ARC candidate disagreement into local
evidence: which regions support the minority candidate and which regions the
vote basin fails.

**A3 ARC mapping:** A3 proved the global set encoder can select; SCOPE tells
.394 how to make the selected or synthesized grid explainable and higher
resolution.

**A4 code mapping:** Code has no comparable per-region evidence source in the
blocked A4 result.

**B2 reward mapping:** Do not train on SCOPE-style pseudo-labels until the B2
harness passes.

**Failure mode:** Fine-grained pseudo-labels can amplify confirmation bias if
they are not tied to exact grid labels.

**Experiment mapping:** Ablate AggLM synthesis with and without SCOPE per-region
evidence on the bigger ARC pool.

## Flagged for .394

`agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394` is the single
strongest method for the next planner. The reason is conditional on the actual
A3/A4/B2 outcomes: Exp 4245 already proved the ARC set-encoder selector beats
vote, Exp 4246 says code robustness is unresolved rather than negative, and
Exp 4248 says reward-weighted SFT is still blocked at the harness gate. So .394
should scale the ARC win with AggLM-style generative reconciliation that
synthesizes a corrected grid from Set-Encoder plus SCOPE per-region evidence on
a bigger pool. Keep code replication as a robustness gate and treat
reward-weighted SFT as an owed gate after the harness proves real training.

