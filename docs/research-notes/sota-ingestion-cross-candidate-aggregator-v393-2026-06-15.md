# SOTA ingestion 2026-06-15: cross-candidate aggregator map for .393

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_cross_candidate_aggregator_mapped_v393`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Set-Encoder full cross-candidate attention`, arxiv_id_or_url: `2404.06912`, url: `https://arxiv.org/abs/2404.06912`}
  - {name: `Calibrated Reasoning explanatory verifier`, arxiv_id_or_url: `2509.19681`, url: `https://arxiv.org/abs/2509.19681`}
  - {name: `Margin-triggered question re-arbitration`, arxiv_id_or_url: `2606.04323`, url: `https://arxiv.org/abs/2606.04323`}
  - {name: `SCOPE fine-grained reward signal`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - {name: `Adaptive verification allocation over categorical structure`, arxiv_id_or_url: `2602.03975`, url: `https://arxiv.org/abs/2602.03975`}
  - {name: `MSV multi-sequence verifier`, arxiv_id_or_url: `2603.03417`, url: `https://arxiv.org/abs/2603.03417`}
  - {name: `AggLM review-reconcile-synthesize aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `AgentAuditor localized branch evidence`, arxiv_id_or_url: `2602.09341`, url: `https://arxiv.org/abs/2602.09341`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v393: `bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner, conditioned on the A2/A3 outcomes.

## Fresh-pass provenance

Read `research-references.md` `.392 planning sweep`, `research-studying.md`,
`results/experiment_4231_oracle_distinct_arc_aggregator_build.json`,
`results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json`, and
`results/experiment_4233_oracle_distinct_code_beats_vote.json`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "permutation invariant inter passage attention listwise reranking verifier candidate set" --limit 8`
- `python3 scripts/sweep_semscholar.py "calibrated reasoning explanatory verifier margin triggered re arbitration multi sequence verifier" --limit 8`

The cluster helper emitted the broadened verifier/process-reward and
energy/verifier arXiv API URLs. Semantic Scholar returned 0 arXiv IDs for the
first focused query and HTTP 429 for the second, so no S2-only promotion is
claimed. Low-concurrency WebSearch/WebFetch verified arXiv:2404.06912,
arXiv:2509.19681, arXiv:2606.04323, arXiv:2512.15146, arXiv:2602.03975,
arXiv:2603.03417, arXiv:2509.06870, and arXiv:2602.09341.

## Exp 4231 A2 build, Exp 4232 ARC A3, and Exp 4233 code read

Exp 4231 did build the strengthened ARC aggregator, but it stayed sparse:
`oracle_distinct_auroc=0.7865558646`, CI95 `[0.6319719028, 0.9258842843]`,
`positive_candidate_n=20`, `wrong_majority_n=9`,
`no_learnable_gain_reason=too_few_positives_after_growth`, and architecture
`cross_candidate_augmented_calibrated_logistic_aggregator`.

Exp 4232 then ran the held-out ARC A3 gate at `held_out_task_n=52` and tied
vote despite headroom: `aggregator_minus_vote_delta=0.0`, CI95 `[0.0, 0.0]`,
`margin_override_minus_vote=0.0`, `oracle_minus_vote=0.1730769231`,
`matched_control_delta=0.0384615385`, and
`oracle_distinct_beats_vote=false`. This is not an under-power n=14 repeat,
but it still leaves the strongest false-negative risk in the sparse-positive
wrong-majority ARC stratum.

Exp 4233 disambiguated the ARC null with code: `code_predictor_minus_vote_delta=0.03125`,
CI95 `[0.00625, 0.0625]`, `held_out_task_n=160`, `code_oracle_distinct_beats_vote=true`,
and `disambiguation_read=ARC_null_is_data_sparsity`. That read is load-bearing:
the next ARC step should grow the ARC pool and change architecture before
declaring the oracle-distinct selection thesis bounded.

## SOTA -> experiment mapping

## Set-Encoder: fix isolated scoring

**Method/source:** Set-Encoder, arXiv:2404.06912
(https://arxiv.org/abs/2404.06912), introduces permutation-invariant
inter-passage attention for listwise reranking.

**Carnot stack mapping:** Replace the Exp 4231 explicit set-statistics
aggregator with a full candidate-set encoder. The model should see all
candidates, vote basins, duplicate families, shape/palette families, and local
grid evidence in one attention pass.

**A2/A3 mapping:** Exp 4232 tied vote after the augmented-feature aggregator.
That makes a full Set-Encoder the strongest architecture lever: it directly
addresses the `.391` isolated scoring cause and tests whether learned
cross-candidate attention beats manual summary features.

**Failure mode:** Set-Encoder is not an ARC solver. If the ARC pool stays at
20 positive candidates and 9 wrong-majority tasks, it may simply learn frequency.

**Experiment mapping:** For .393, grow the ARC pool first, then compare isolated
scoring, Exp 4231 summary features, and a full Set-Encoder on identical
task-held-out splits.

## Calibrated Reasoning: fix class imbalance

**Method/source:** Calibrated Reasoning, arXiv:2509.19681
(https://arxiv.org/abs/2509.19681), trains an explanatory verifier with
calibrated confidence for candidate solutions.

**Carnot stack mapping:** Keep the class-balanced/calibrated loss, but make it
auditable: report score histograms, positive/negative calibration, and
wrong-majority margins rather than only AUROC.

**A2/A3 mapping:** Exp 4231 used a balanced logistic objective and isotonic
calibration, yet still had too few positives after growth. Exp 4233 won on a
larger, less sparse code pool, so the .393 calibration step needs more ARC data,
not just another loss variant.

**Failure mode:** Calibration cannot create missing positives or local grid
evidence. A better loss on the same sparse pool risks a cleaner-looking tie.

**Experiment mapping:** Pair the bigger ARC pool with class-weighted, focal,
and pairwise calibrated losses, then report whether any loss creates nonzero
wrong-majority margins for the Set-Encoder.

## Margin-triggered re-arbitration: fix override degeneracy

**Method/source:** Margin-triggered question re-arbitration, arXiv:2606.04323
(https://arxiv.org/abs/2606.04323), conditions re-arbitration on the
self-consistency vote margin.

**Carnot stack mapping:** Use the margin trigger as the final deployment guard:
keep vote unless the learned score margin over vote clears a pre-registered
threshold.

**A2/A3 mapping:** Exp 4232's margin override tied vote too, with
`margin_override_minus_vote=0.0`. Therefore margin-triggering should stay as an
evaluation arm, not become the .393 headline by itself.

**Failure mode:** A margin policy with no meaningful score separation either
never fires or fires on noise. The cited paper also reports sensitivity to the
triggered subset.

**Experiment mapping:** Retain the fixed margin-trigger policy after the
Set-Encoder produces margins; report selector@1, margin override, vote, and
matched control.

## SCOPE: add fine-grained reward and per-region evidence

**Method/source:** SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), moves beyond majority voting with
step-wise confidence and subgroup-specific pseudo-label estimation.

**Carnot stack mapping:** Convert ARC grid disagreement into per-region evidence
features so the model has a reason to trust a minority answer.

**A2/A3 mapping:** Exp 4232 left `oracle_minus_vote=0.1730769231` on the table.
SCOPE maps to denser evidence for those tasks, not to a new majority-vote
threshold.

**Failure mode:** SCOPE is not an ARC label generator. Without exact grid labels
and region evidence, it could amplify pseudo-label bias.

**Experiment mapping:** Add region disagreement features before verifier-as-reward
training; ablate Set-Encoder with and without SCOPE-style local evidence.

## Adaptive verification allocation: route scarce checks

**Method/source:** Adaptive verification allocation, arXiv:2602.03975
(https://arxiv.org/abs/2602.03975), allocates costly verification over
structured intermediate states.

**Carnot stack mapping:** Route evidence gathering to uncertain ARC candidate
families, especially duplicate grids and high-uncertainty transformation
families.

**A2/A3 mapping:** Exp 4232's matched control remained close enough that .393
should separate score quality from compute routing.

**Failure mode:** Allocation is an efficiency lever, not the primary
vote-beating mechanism. It cannot rescue an uninformative score.

**Experiment mapping:** Make adaptive allocation a secondary arm after the full
Set-Encoder baseline exists.

## MSV: jointly process candidate solutions

**Method/source:** Multi-Sequence Verifier, arXiv:2603.03417
(https://arxiv.org/abs/2603.03417), jointly processes candidate solutions and
models their interactions.

**Carnot stack mapping:** Treat ARC candidates as a set and calibrate scores
with cross-candidate interactions, rather than independent candidate rows.

**A2/A3 mapping:** MSV corroborates the Set-Encoder diagnosis: the Exp 4231
summary-feature aggregator was only a proxy for direct cross-sequence modeling.

**Failure mode:** Cross-sequence calibration can still track frequency without
better positive support.

**Experiment mapping:** Report three arms: isolated scoring, explicit summary
features, and full multi-sequence/set attention on the expanded ARC pool.

## AggLM and AgentAuditor: synthesize or audit evidence

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), learns to review, reconcile, and synthesize
answers. AgentAuditor, arXiv:2602.09341
(https://arxiv.org/abs/2602.09341), audits localized reasoning-tree branch
evidence and targets majority-failure cases.

**Carnot stack mapping:** If selector-only Set-Encoder still ties, the next arm
should synthesize a corrected grid from candidate evidence or audit the
localized evidence behind the minority candidate.

**A2/A3 mapping:** Exp 4232 shows a selector tie, while Exp 4233 says the thesis
is not bounded in a higher-power domain. Therefore .393 should not stop at
selection-only reranking if a bigger ARC pool still leaves oracle headroom.

**Failure mode:** Synthesis increases fabrication risk, and AgentAuditor assumes
richer trace evidence than cached ARC rows may contain.

**Experiment mapping:** Add an AggLM-style synthesize corrected grid arm after
the full Set-Encoder baseline. Keep exact grid-match validation and compare
against an AgentAuditor localized-evidence audit and LLM-as-judge fallback.

## Flagged for .393

`bigger_arc_pool_full_set_encoder_agglm_aggregator_v393` is the strongest single
method for the next planner. The reason is conditional on the actual A2/A3
outcomes: ARC tied vote with headroom, but the higher-power code read beat vote
and explicitly reported `ARC_null_is_data_sparsity`. The .393 plan should grow
the ARC pool, run a full Set-Encoder against the augmented-feature aggregator,
and reserve AggLM-style synthesis for the case where selection still leaves
oracle headroom unused. Build a bigger ARC pool before declaring the
oracle-distinct selection thesis bounded.

